"""Tests for the experimental staffing agent.

The agent is allowed to be bad -- and on the pre-declared training budget it is. What
is not allowed is a broken environment, a reward that lets high-acuity waiting be
traded for money, or an evaluation that scores the agent on the worlds it trained on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.simulation.calibrate import calibrate_and_validate
from src.simulation.staffing_rl import (
    EXPERIMENTAL_NOTICE,
    EnvConfig,
    FixedRosterPolicy,
    QLearningPolicy,
    RandomPolicy,
    ReactiveRulePolicy,
    RewardWeights,
    StaffingEnv,
    compare_policies,
    evaluate_policy,
    train_q_learning,
)


@pytest.fixture(scope="module")
def calibration(default_config: dict, tiny_visits_module):
    """A calibration fitted on the small fixture, for fast tests."""
    import copy

    config = copy.deepcopy(default_config)
    config["simulation"]["calibration"]["replications"] = 2
    return calibrate_and_validate(tiny_visits_module, config, n_replications=2)


@pytest.fixture(scope="module")
def tiny_visits_module(default_config: dict) -> pd.DataFrame:
    """Validated visits from the small synthetic fixture."""
    import copy

    from src.data.schema import load_schema, validate_visits
    from src.data.synthetic import generate_synthetic_visits

    config = copy.deepcopy(default_config)
    config["synthetic"]["n_patients"] = 4000
    return validate_visits(generate_synthetic_visits(config), load_schema()).all


@pytest.fixture()
def env(default_config: dict, calibration) -> StaffingEnv:
    """A short-episode environment."""
    return StaffingEnv(
        default_config, calibration, env_config=EnvConfig(hours=48, warmup_hours=6)
    )


# ---------------------------------------------------------------- the environment


def test_the_environment_steps_an_hour_at_a_time(env: StaffingEnv) -> None:
    """One step is one hour, and the episode ends when the hours run out."""
    env.reset("test::episode")
    steps = 0
    done = False
    while not done:
        _state, _reward, done, info = env.step(StaffingEnv.ACTIONS.index(0))
        steps += 1
        assert info["hour"] == steps

    assert steps == env.env_config.hours


def test_actions_cannot_leave_the_allowed_range(env: StaffingEnv) -> None:
    """Staffing is bounded; the agent must not be able to escape the bounds."""
    env.reset("test::bounds")
    for _ in range(30):
        _state, _reward, _done, info = env.step(StaffingEnv.ACTIONS.index(-1))
        assert info["doctors"] >= env.env_config.min_doctors

    env.reset("test::bounds_up")
    for _ in range(30):
        _state, _reward, _done, info = env.step(StaffingEnv.ACTIONS.index(1))
        assert info["doctors"] <= env.env_config.max_doctors


def test_the_state_index_is_inside_the_table(env: StaffingEnv) -> None:
    """A tabular agent indexes an array with this; out of range is a crash."""
    state = env.reset("test::state")
    assert 0 <= state < env.n_states

    done = False
    while not done:
        state, _reward, done, _info = env.step(StaffingEnv.ACTIONS.index(1))
        assert 0 <= state < env.n_states


def test_the_same_seed_label_gives_the_same_world(env: StaffingEnv) -> None:
    """Common random numbers: without this the policy comparison is noise."""
    def run() -> list[float]:
        env.reset("test::crn")
        rewards = []
        done = False
        while not done:
            _state, reward, done, _info = env.step(StaffingEnv.ACTIONS.index(0))
            rewards.append(reward)
        return rewards

    assert run() == run()


def test_more_doctors_cannot_increase_waiting(env: StaffingEnv) -> None:
    """A sign check on the environment, not on any policy."""
    def total_wait_penalty(action: int) -> float:
        env.reset("test::monotone")
        total = 0.0
        done = False
        while not done:
            _state, _reward, done, info = env.step(action)
            total += info["wait_penalty"]
        return total

    assert total_wait_penalty(StaffingEnv.ACTIONS.index(1)) <= total_wait_penalty(
        StaffingEnv.ACTIONS.index(-1)
    )


# ---------------------------------------------------------------- the reward


def test_a_high_acuity_breach_costs_far_more_than_a_doctor_hour() -> None:
    """The reward must not let ESI 1-2 waiting be traded away for money."""
    weights = RewardWeights()

    assert weights.high_acuity_breach_penalty > 4 * weights.cost_per_doctor_hour


def test_staffing_cost_scales_with_doctors(env: StaffingEnv) -> None:
    """The money side of the reward, checked directly."""
    env.reset("test::cost")
    _state, _reward, _done, low = env.step(StaffingEnv.ACTIONS.index(-1))
    cheap = low["cost"]
    for _ in range(4):
        _state, _reward, _done, high = env.step(StaffingEnv.ACTIONS.index(1))

    assert high["cost"] > cheap


def test_the_reward_is_never_positive(env: StaffingEnv) -> None:
    """It is a cost, negated. A positive reward would mean staffing earned money."""
    env.reset("test::sign")
    done = False
    while not done:
        _state, reward, done, _info = env.step(StaffingEnv.ACTIONS.index(0))
        assert reward <= 0.0


# ---------------------------------------------------------------- the policies


def test_the_fixed_policy_converges_on_the_roster(env: StaffingEnv, calibration) -> None:
    """The incumbent must actually hold the rostered level."""
    policy = FixedRosterPolicy(calibration.plan.capacity_by_hour(), env.env_config)
    state = env.reset("test::fixed")
    for _ in range(12):
        state, _reward, _done, info = env.step(policy.act(state, env))

    rostered = calibration.plan.capacity_by_hour()[info["hour"] % 24]
    assert abs(info["doctors"] - rostered) <= 1


def test_the_random_policy_is_reproducible(env: StaffingEnv) -> None:
    """Even the floor baseline has to be re-runnable."""
    def run() -> list[int]:
        policy = RandomPolicy(seed=7)
        state = env.reset("test::random")
        chosen = []
        for _ in range(10):
            action = policy.act(state, env)
            chosen.append(action)
            state, _r, _d, _i = env.step(action)
        return chosen

    assert run() == run()


def test_a_q_policy_is_greedy_when_epsilon_is_zero(env: StaffingEnv) -> None:
    """A trained policy must be deterministic, or evaluation means nothing."""
    table = np.zeros((env.n_states, env.n_actions))
    table[:, 2] = 1.0  # always prefer "add a doctor"
    policy = QLearningPolicy(q_table=table, epsilon=0.0)

    state = env.reset("test::greedy")
    assert policy.act(state, env) == 2


# ---------------------------------------------------------------- the comparison


def test_the_comparison_is_paired_on_episode() -> None:
    """Unpaired, the policies' shared random numbers would be thrown away."""
    results = pd.DataFrame(
        [
            {"policy": "fixed_roster", "episode": 0, "total_cost": 100.0,
             "staffing_cost": 80.0, "high_acuity_breaches": 1},
            {"policy": "fixed_roster", "episode": 1, "total_cost": 300.0,
             "staffing_cost": 80.0, "high_acuity_breaches": 1},
            {"policy": "q_learning", "episode": 0, "total_cost": 90.0,
             "staffing_cost": 70.0, "high_acuity_breaches": 0},
            {"policy": "q_learning", "episode": 1, "total_cost": 290.0,
             "staffing_cost": 70.0, "high_acuity_breaches": 0},
        ]
    )
    comparison = compare_policies(results).set_index("policy")

    # Both episodes differ by exactly -10, so the paired difference is -10 with a
    # zero-width interval, even though the raw costs span 100 to 300.
    assert comparison.loc["q_learning", "cost_difference"] == pytest.approx(-10.0)
    assert comparison.loc["q_learning", "beats_baseline"] is True


def test_the_comparison_refuses_without_a_baseline() -> None:
    """Differencing against nothing would silently produce an empty table."""
    with pytest.raises(ValueError, match="baseline"):
        compare_policies(
            pd.DataFrame([{"policy": "q_learning", "episode": 0, "total_cost": 1.0,
                           "staffing_cost": 1.0, "high_acuity_breaches": 0}])
        )


def test_every_comparison_carries_the_experimental_notice() -> None:
    """Nothing from this module may be read as a recommendation."""
    results = pd.DataFrame(
        [
            {"policy": "fixed_roster", "episode": 0, "total_cost": 1.0,
             "staffing_cost": 1.0, "high_acuity_breaches": 0},
        ]
    )
    assert (compare_policies(results)["notice"] == EXPERIMENTAL_NOTICE).all()


def test_the_notice_names_the_circularity() -> None:
    """The caveat has to say what is actually wrong with the result."""
    assert "simulator" in EXPERIMENTAL_NOTICE
    assert "synthetic" in EXPERIMENTAL_NOTICE
    assert "not evidence" in EXPERIMENTAL_NOTICE


# ---------------------------------------------------------------- training


@pytest.mark.slow
def test_training_runs_and_produces_a_usable_policy(env: StaffingEnv) -> None:
    """A short run, checking the machinery rather than the outcome."""
    policy, log = train_q_learning(env, episodes=3, seed=1, seed_prefix="test::train")

    assert policy.epsilon == 0.0, "a trained policy must be greedy"
    assert len(log) == 3
    assert policy.q_table.shape == (env.n_states, env.n_actions)
    assert np.isfinite(policy.q_table).all()


@pytest.mark.slow
def test_training_and_evaluation_use_different_worlds(env: StaffingEnv) -> None:
    """Scoring an agent on its training episodes would make any win meaningless."""
    policy, _log = train_q_learning(env, episodes=2, seed=1, seed_prefix="rl::train")
    results = evaluate_policy(env, policy, seeds=[0, 1], seed_prefix="rl::eval")

    assert len(results) == 2
    assert (results["policy"] == "q_learning").all()
