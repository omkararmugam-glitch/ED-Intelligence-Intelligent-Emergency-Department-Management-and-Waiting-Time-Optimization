"""An hourly staffing policy, learned against the simulator. Experimental.

**Read the circularity before reading any result.**

The agent learns to staff a SimPy model. That model was calibrated to reproduce
`data/raw/ed_v2.csv`, which was produced by a generator whose parameters were chosen
by hand. So an agent that beats the fixed roster has demonstrated that the *method*
works on a queueing system whose dynamics it was allowed to learn by playing it
millions of times. It has demonstrated nothing whatever about a real emergency
department, where the dynamics are unknown, non-stationary, and not available to play
against.

Three specific things make the win easier here than it would ever be in practice:

1. **The environment is the model.** There is no sim-to-real gap because there is no
   real. Every irregularity a real department has -- a broken scanner, a nurse off
   sick, an ambulance diversion -- is absent by construction.
2. **The agent sees the true queue state.** A real system sees a tracking board that
   is minutes stale and sometimes wrong.
3. **Staffing is instantly adjustable.** The action "add a doctor for the next hour"
   has no notice period, no roster, no overtime rules and no one to ask.

If the learned policy does not beat the fixed plan, that is reported as it stands.
Nothing here is tuned until it wins.

**Dependencies.** Tabular Q-learning, implemented on NumPy. No gymnasium, no torch, no
RL library: the interface below is a few methods, and a dependency that must be
installed before the rest of the project works would be a worse trade than writing
them. `src.simulation.staffing_rl` imports nothing the project does not already use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.simulation.calibrate import CalibrationResult
from src.simulation.des import MINUTES_PER_HOUR, EDSimulation
from src.utils.logging import get_logger
from src.utils.metrics import mean_ci
from src.utils.seed import rng as make_rng

logger = get_logger(__name__)

#: Carried into every artifact this module writes.
EXPERIMENTAL_NOTICE = (
    "EXPERIMENTAL. This policy was learned against the project's own simulator, which "
    "was calibrated to synthetic data. A win here shows the method runs end to end; it "
    "is not evidence that the policy would help a real emergency department, where the "
    "dynamics are unknown and cannot be played against millions of times."
)


@dataclass(frozen=True)
class RewardWeights:
    """What the agent is being asked to trade off.

    Attributes:
        cost_per_doctor_hour: Money, matching `staffing_search.cost.doctor_hour`.
        target_wait_minutes: The wait above which ESI 3-5 minutes start costing.
        penalty_per_wait_minute: Cost of one patient-minute over that target.
        high_acuity_target_minutes: The ESI 1-2 wait that counts as a breach.
        high_acuity_breach_penalty: Cost of one ESI 1-2 breach. Large on purpose: a
            high-acuity patient waiting is the failure this department exists to
            avoid, and a reward function that let the agent trade it for money would
            be the wrong reward function whatever it then learned.
    """

    cost_per_doctor_hour: float = 110.0
    target_wait_minutes: float = 60.0
    penalty_per_wait_minute: float = 1.0
    high_acuity_target_minutes: float = 15.0
    high_acuity_breach_penalty: float = 500.0


@dataclass(frozen=True)
class EnvConfig:
    """The decision problem's shape."""

    min_doctors: int = 1
    max_doctors: int = 6
    hours: int = 168
    warmup_hours: int = 24
    #: Queue-length bucket edges, per acuity band, for the tabular state.
    queue_buckets: tuple[int, ...] = (0, 3, 8, 16, 32)
    #: Hour-of-day bands, which is all the time signal a tabular agent can use.
    hour_bands: int = 4


class StaffingEnv:
    """The SimPy department, stepped one hour at a time.

    A deliberately small interface -- `reset`, `step`, `state_index`, `n_actions` --
    rather than a gymnasium dependency. It is the same shape gymnasium would give.

    One step is one hour: the agent sets the doctor count, the simulation runs that
    hour, and the reward is that hour's cost plus the waiting it caused.
    """

    #: -1, 0 or +1 doctors relative to the current level.
    ACTIONS = (-1, 0, 1)

    def __init__(
        self,
        config: dict[str, Any],
        calibration: CalibrationResult,
        env_config: EnvConfig | None = None,
        rewards: RewardWeights | None = None,
    ) -> None:
        self.config = config
        self.calibration = calibration
        self.env_config = env_config or EnvConfig()
        self.rewards = rewards or RewardWeights(
            cost_per_doctor_hour=float(
                config.get("staffing_search", {}).get("cost", {}).get("doctor_hour", 110.0)
            ),
            target_wait_minutes=float(
                config.get("staffing_search", {})
                .get("target", {})
                .get("p90_wait_esi3_minutes", 90.0)
            ),
        )
        self.simulation: EDSimulation | None = None
        self.doctors = 0
        self.hour = 0
        self._scored_from = 0

    # ---------------------------------------------------------------- the interface

    @property
    def n_actions(self) -> int:
        """How many actions the agent may take."""
        return len(self.ACTIONS)

    @property
    def n_states(self) -> int:
        """Size of the discretised state space."""
        buckets = len(self.env_config.queue_buckets)
        doctors = self.env_config.max_doctors - self.env_config.min_doctors + 1
        return buckets * buckets * doctors * self.env_config.hour_bands

    def reset(self, seed_label: str) -> int:
        """Start a fresh episode and return the initial state index.

        Args:
            seed_label: Names the replication's random stream. Passing the same label
                to two policies gives them common random numbers -- the same arrivals
                and the same service draws -- which is what makes the comparison a
                comparison of policies rather than of luck.

        Returns:
            The initial discretised state.
        """
        horizon = self.env_config.hours
        self.simulation = EDSimulation(
            config=self.config,
            arrivals=self.calibration.arrivals,
            service=self.calibration.service,
            plan=self.calibration.plan,
            seed_label=seed_label,
        )
        self.doctors = int(
            np.clip(
                self.calibration.plan.capacity_by_hour()[0],
                self.env_config.min_doctors,
                self.env_config.max_doctors,
            )
        )
        self.simulation.dynamic_capacity = self.doctors
        self.simulation.env.process(self.simulation._arrival_process())
        self.simulation.env.process(self.simulation._shift_clock())
        self.simulation.env.process(self.simulation._monitor())
        self.hour = 0
        self._scored_from = 0
        return self.state_index()

    def step(self, action: int) -> tuple[int, float, bool, dict[str, Any]]:
        """Apply one hour of staffing and advance the simulation by an hour.

        Args:
            action: Index into `ACTIONS`.

        Returns:
            Tuple of (next state, reward, done, info).
        """
        assert self.simulation is not None, "call reset() first"

        self.doctors = int(
            np.clip(
                self.doctors + self.ACTIONS[action],
                self.env_config.min_doctors,
                self.env_config.max_doctors,
            )
        )
        self.simulation.dynamic_capacity = self.doctors

        before = len(self.simulation.patients)
        self.hour += 1
        self.simulation.env.run(until=self.hour * MINUTES_PER_HOUR)

        reward, info = self._reward(before)
        done = self.hour >= self.env_config.hours
        info.update(hour=self.hour, doctors=self.doctors)
        return self.state_index(), reward, done, info

    # ---------------------------------------------------------------- internals

    def _waiting_by_band(self) -> tuple[int, int]:
        """How many ESI 1-2 and ESI 3-5 patients are waiting right now."""
        assert self.simulation is not None
        high = low = 0
        for index in self.simulation.waiting:
            if self.simulation.patients[index].triage <= 2:
                high += 1
            else:
                low += 1
        return high, low

    def _reward(self, patients_before: int) -> tuple[float, dict[str, Any]]:
        """Cost of this hour's staffing, plus the waiting it allowed.

        Waiting is charged on patients *seen during this hour*, so the reward is
        attributed to the hour whose staffing produced it rather than to the hour the
        patient arrived in.
        """
        assert self.simulation is not None
        start = (self.hour - 1) * MINUTES_PER_HOUR
        end = self.hour * MINUTES_PER_HOUR

        over_target = 0.0
        breaches = 0
        seen_this_hour = 0
        for patient in self.simulation.patients:
            if patient.seen_at is None or not (start <= patient.seen_at < end):
                continue
            seen_this_hour += 1
            wait = patient.seen_at - patient.arrival
            if patient.triage <= 2:
                if wait > self.rewards.high_acuity_target_minutes:
                    breaches += 1
            elif wait > self.rewards.target_wait_minutes:
                over_target += wait - self.rewards.target_wait_minutes

        cost = self.doctors * self.rewards.cost_per_doctor_hour
        penalty = over_target * self.rewards.penalty_per_wait_minute
        breach_cost = breaches * self.rewards.high_acuity_breach_penalty
        reward = -(cost + penalty + breach_cost)

        return reward, {
            "cost": cost,
            "wait_penalty": penalty,
            "breach_penalty": breach_cost,
            "breaches": breaches,
            "minutes_over_target": over_target,
            "seen_this_hour": seen_this_hour,
            "scored": self.hour > self.env_config.warmup_hours,
        }

    def state_index(self) -> int:
        """Discretise the current state for a tabular agent.

        Queue length by acuity band, doctors on duty and the hour-of-day band: the
        state the brief asks for, bucketed because a tabular agent needs a finite one.
        """
        assert self.simulation is not None
        high, low = self._waiting_by_band()
        buckets = self.env_config.queue_buckets
        high_bucket = int(np.searchsorted(buckets, high, side="right") - 1)
        low_bucket = int(np.searchsorted(buckets, low, side="right") - 1)
        doctor_index = self.doctors - self.env_config.min_doctors
        band = int((self.hour % 24) // (24 // self.env_config.hour_bands))

        n_buckets = len(buckets)
        n_doctors = self.env_config.max_doctors - self.env_config.min_doctors + 1
        return int(
            ((high_bucket * n_buckets + low_bucket) * n_doctors + doctor_index)
            * self.env_config.hour_bands
            + band
        )


# ------------------------------------------------------------------ the policies


class Policy:
    """Anything that can choose an action. Three baselines and one learned."""

    name = "policy"
    description = ""

    def act(self, state: int, env: StaffingEnv) -> int:
        """Choose an action index."""
        raise NotImplementedError


class FixedRosterPolicy(Policy):
    """Do nothing: hold the rostered doctor count, whatever the queue does.

    This is the incumbent -- the `day=3, evening=3, night_early=2, night_late=2` plan
    the staffing search selects at a cost of 49,280 per week. Beating it is the bar.
    """

    name = "fixed_roster"
    description = "hold the rostered level (the cheapest feasible plan)"

    def __init__(self, capacity_by_hour: np.ndarray, env_config: EnvConfig) -> None:
        self.capacity_by_hour = capacity_by_hour
        self.env_config = env_config

    def act(self, state: int, env: StaffingEnv) -> int:
        """Move one step towards the rostered level for this hour of day."""
        target = int(
            np.clip(
                self.capacity_by_hour[env.hour % 24],
                self.env_config.min_doctors,
                self.env_config.max_doctors,
            )
        )
        if env.doctors < target:
            return StaffingEnv.ACTIONS.index(1)
        if env.doctors > target:
            return StaffingEnv.ACTIONS.index(-1)
        return StaffingEnv.ACTIONS.index(0)


class ReactiveRulePolicy(Policy):
    """A rule a charge nurse could follow: staff up when the queue is long.

    Written by hand rather than learned, and deliberately simple -- it is the "would
    anyone bother with RL" control. If the learned policy cannot beat this, the honest
    conclusion is to use this.
    """

    name = "reactive_rule"
    description = "add a doctor when more than 8 are waiting, drop one below 3"

    def __init__(self, add_above: int = 8, drop_below: int = 3) -> None:
        self.add_above = add_above
        self.drop_below = drop_below

    def act(self, state: int, env: StaffingEnv) -> int:
        """Up if the queue is long, down if it is short, otherwise hold."""
        high, low = env._waiting_by_band()
        waiting = high + low
        if high > 0 or waiting > self.add_above:
            return StaffingEnv.ACTIONS.index(1)
        if waiting < self.drop_below:
            return StaffingEnv.ACTIONS.index(-1)
        return StaffingEnv.ACTIONS.index(0)


class RandomPolicy(Policy):
    """Uniformly random. The floor: anything that cannot beat this is broken."""

    name = "random"
    description = "uniformly random action each hour"

    def __init__(self, seed: int = 0) -> None:
        self.rng = make_rng(seed, "rl::random_policy")

    def act(self, state: int, env: StaffingEnv) -> int:
        """Pick an action at random."""
        return int(self.rng.integers(0, env.n_actions))


@dataclass
class QLearningPolicy(Policy):
    """Tabular Q-learning over the discretised state.

    Tabular rather than a DQN, chosen for stability: the state space here is a few
    hundred cells, a table converges without a replay buffer, a target network or any
    of the machinery that makes a small DQN fiddly to get right. The brief asked for
    whichever was simpler to make stable, and this is unambiguously it.
    """

    q_table: np.ndarray
    epsilon: float = 0.0
    name: str = "q_learning"
    description: str = "tabular Q-learning over (queue by acuity, doctors, hour band)"
    rng: Any = None

    def act(self, state: int, env: StaffingEnv) -> int:
        """Greedy, or epsilon-greedy while training."""
        if self.epsilon > 0 and self.rng is not None and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, env.n_actions))
        return int(np.argmax(self.q_table[state]))


# ------------------------------------------------------------------ training


def train_q_learning(
    env: StaffingEnv,
    episodes: int = 60,
    alpha: float = 0.2,
    gamma: float = 0.95,
    epsilon_start: float = 0.5,
    epsilon_end: float = 0.02,
    seed: int = 0,
    seed_prefix: str = "rl::train",
) -> tuple[QLearningPolicy, pd.DataFrame]:
    """Train a tabular agent, on episodes disjoint from the evaluation ones.

    Args:
        env: The environment.
        episodes: Training episodes.
        alpha: Learning rate.
        gamma: Discount.
        epsilon_start: Initial exploration rate, annealed linearly.
        epsilon_end: Final exploration rate.
        seed: Seed for exploration.
        seed_prefix: Prefix for the training episodes' random streams. It must differ
            from the evaluation prefix, or the agent is tested on worlds it trained on.

    Returns:
        Tuple of (greedy policy, per-episode training log).
    """
    rng = make_rng(seed, "rl::q_learning")
    q_table = np.zeros((env.n_states, env.n_actions), dtype="float64")
    policy = QLearningPolicy(q_table=q_table, epsilon=epsilon_start, rng=rng)

    rows: list[dict[str, Any]] = []
    for episode in range(episodes):
        policy.epsilon = epsilon_start + (epsilon_end - epsilon_start) * (
            episode / max(episodes - 1, 1)
        )
        state = env.reset(f"{seed_prefix}::{episode}")
        total = 0.0
        breaches = 0
        done = False
        while not done:
            action = policy.act(state, env)
            next_state, reward, done, info = env.step(action)
            best_next = float(np.max(q_table[next_state]))
            q_table[state, action] += alpha * (
                reward + gamma * best_next - q_table[state, action]
            )
            state = next_state
            if info["scored"]:
                total += reward
                breaches += info["breaches"]
        rows.append(
            {"episode": episode, "epsilon": round(policy.epsilon, 4),
             "total_reward": round(total, 2), "breaches": breaches}
        )
        if episode % 10 == 0 or episode == episodes - 1:
            logger.info(
                "Q-learning episode %d/%d: reward %.0f, %d high-acuity breach(es), "
                "epsilon %.3f",
                episode + 1, episodes, total, breaches, policy.epsilon,
            )

    policy.epsilon = 0.0  # greedy from here on
    return policy, pd.DataFrame(rows)


# ------------------------------------------------------------------ evaluation


def evaluate_policy(
    env: StaffingEnv, policy: Policy, seeds: Sequence[int], seed_prefix: str = "rl::eval"
) -> pd.DataFrame:
    """Run one policy over the evaluation seeds and record what it cost.

    Every policy is given the same `seed_prefix` and the same seeds, so episode *i* of
    each sees the same arrivals and service draws. That is common random numbers, and
    it is what lets a handful of episodes separate policies that differ by a few
    percent.

    Args:
        env: The environment.
        policy: The policy to evaluate.
        seeds: Evaluation episode indices.
        seed_prefix: Shared across policies.

    Returns:
        One row per episode.
    """
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        state = env.reset(f"{seed_prefix}::{seed}")
        cost = penalty = breach_cost = 0.0
        breaches = 0
        doctor_hours = 0
        done = False
        while not done:
            state, _reward, done, info = env.step(policy.act(state, env))
            if not info["scored"]:
                continue
            cost += info["cost"]
            penalty += info["wait_penalty"]
            breach_cost += info["breach_penalty"]
            breaches += info["breaches"]
            doctor_hours += info["doctors"]
        rows.append(
            {
                "policy": policy.name,
                "episode": seed,
                "staffing_cost": round(cost, 2),
                "wait_penalty": round(penalty, 2),
                "breach_penalty": round(breach_cost, 2),
                "total_cost": round(cost + penalty + breach_cost, 2),
                "high_acuity_breaches": breaches,
                "doctor_hours": doctor_hours,
            }
        )
    return pd.DataFrame(rows)


def compare_policies(
    results: pd.DataFrame, baseline: str = "fixed_roster", confidence: float = 0.95
) -> pd.DataFrame:
    """Paired comparison of every policy against the incumbent roster.

    Paired on episode, because the policies share random numbers. The interval is on
    the mean per-episode difference, which is the quantity that answers "is this policy
    better", rather than on the difference of two noisy means.

    Args:
        results: Concatenated output of `evaluate_policy`.
        baseline: The policy to difference against.
        confidence: Interval level.

    Returns:
        One row per policy, with the paired difference and whether it is significant.
    """
    reference = results[results["policy"] == baseline]
    if reference.empty:
        raise ValueError(f"No rows for the baseline policy {baseline!r}")

    rows: list[dict[str, Any]] = []
    for name, group in results.groupby("policy", sort=False):
        merged = group.merge(reference, on="episode", suffixes=("", "_base"))
        row: dict[str, Any] = {
            "policy": name,
            "n_episodes": int(len(merged)),
            "mean_total_cost": round(float(merged["total_cost"].mean()), 2),
            "mean_staffing_cost": round(float(merged["staffing_cost"].mean()), 2),
            "mean_breaches": round(float(merged["high_acuity_breaches"].mean()), 3),
        }
        if name != baseline:
            difference = merged["total_cost"] - merged["total_cost_base"]
            stats = mean_ci(difference, confidence=confidence)
            row.update(
                cost_difference=round(float(stats["mean"]), 2),
                ci_low=round(float(stats["ci_lower"]), 2),
                ci_high=round(float(stats["ci_upper"]), 2),
                beats_baseline=bool(stats["mean"] < 0),
                significant=bool(stats["ci_upper"] < 0 or stats["ci_lower"] > 0),
            )
            breach_difference = (
                merged["high_acuity_breaches"] - merged["high_acuity_breaches_base"]
            )
            row["breach_difference"] = round(float(breach_difference.mean()), 3)
        rows.append(row)

    table = pd.DataFrame(rows).sort_values("mean_total_cost", ignore_index=True)
    table["notice"] = EXPERIMENTAL_NOTICE
    return table
