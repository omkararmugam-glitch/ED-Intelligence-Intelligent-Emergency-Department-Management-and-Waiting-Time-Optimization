"""
Realism-calibrated SYNTHETIC emergency department dataset (v2).

NOT real patient data. Parameters are tuned toward published benchmark ranges
(approximate: NHAMCS-style median wait ~30 min, ESI-1 median wait well under 15 min,
median treatment time ~90 min, LWBS in the low single digits). Hourly/weekly arrival
shape is a hand-set typical ED profile, not fitted to a specific hospital.

Usage:  python generate_ed_data.py --days 365 --seed 42 --out ed_v2.csv
"""
import argparse
import heapq
import numpy as np
import pandas as pd

ESI_MIX = np.array([0.02, 0.12, 0.40, 0.33, 0.13])

HOUR_W = np.array([0.45, 0.36, 0.30, 0.26, 0.24, 0.25, 0.32, 0.45, 0.62, 0.78, 0.90, 0.97,
                   1.00, 0.98, 0.95, 0.93, 0.93, 0.95, 0.97, 0.95, 0.85, 0.72, 0.62, 0.52])
DOW_W = np.array([1.08, 1.00, 0.98, 0.97, 0.98, 0.93, 0.96])  # Mon..Sun

# physician-busy time per patient (minutes, lognormal median, sigma) by ESI
CONSULT_MED = {1: 32, 2: 26, 3: 20, 4: 12, 5: 8}
CONSULT_SIG = 0.55
# total time from first physician contact to departure, by disposition (median, sigma)
LOS_MED = {"discharged": 85, "admitted": 190, "transferred": 120}
LOS_SIG = 0.55
ADMIT_P = {1: 0.80, 2: 0.55, 3: 0.25, 4: 0.05, 5: 0.02}
TRANSFER_P = 0.012
# patience (minutes) before leaving without being seen; ESI 1-2 never leave
PATIENCE_MED = {3: 420, 4: 240, 5: 200}
PATIENCE_SIG = 0.8
AGING_MIN, AGING_FLOOR = 25.0, 2.5  # same aging rule family as the project's generator

AMBULANCE_P = {1: 0.70, 2: 0.40, 3: 0.20, 4: 0.07, 5: 0.03}
COMPLAINTS = {
    1: ["cardiac arrest", "respiratory failure", "major trauma", "unresponsive"],
    2: ["chest pain", "stroke symptoms", "severe shortness of breath", "sepsis concern", "major bleeding"],
    3: ["abdominal pain", "chest pain", "fracture", "shortness of breath", "fever", "headache", "laceration"],
    4: ["minor injury", "sprain", "urinary symptoms", "sore throat", "back pain", "rash", "laceration"],
    5: ["medication refill", "minor rash", "cold symptoms", "suture removal", "insect bite"],
}


def n_doctors(hour):
    # staffing by hour of day (tuned in calibrate loop)
    if hour < 7:
        return N_NIGHT
    if hour < 15:
        return N_DAY
    if hour < 23:
        return N_EVE
    return N_NIGHT


N_NIGHT, N_DAY, N_EVE = 2, 3, 3
MAX_DOC = 6


def slot_active(slot, t_min):
    hour = int((t_min // 60) % 24)
    return slot < n_doctors(hour)


def next_active(slot, t_min):
    t = t_min
    for _ in range(48):
        if slot_active(slot, t):
            return t
        t = (t // 60 + 1) * 60
    return t


def gen_arrivals(rng, days, scale):
    base_rate = scale  # mean arrivals/hour at weight 1.0
    times = []
    for d in range(days):
        dow = d % 7
        season = 1.0 + 0.06 * np.sin(2 * np.pi * d / 365.0 + 1.0)
        day_factor = rng.lognormal(0, 0.08) * (1.35 if rng.random() < 0.015 else 1.0)  # random surge days
        for h in range(24):
            lam = base_rate * HOUR_W[h] * DOW_W[dow] * season * day_factor
            k = rng.poisson(lam)
            if k:
                times.extend(d * 1440 + h * 60 + rng.uniform(0, 60, size=k))
    return np.sort(np.array(times))


def draw_patients(rng, n):
    age = np.clip(np.round(np.where(rng.random(n) < 0.22, rng.normal(72, 10, n),
                                    np.where(rng.random(n) < 0.18, rng.normal(8, 5, n), rng.normal(42, 17, n)))), 0, 98).astype(int)
    # older patients slightly more acute
    mix = np.tile(ESI_MIX, (n, 1))
    boost = np.clip((age - 60) / 60, 0, 0.5)
    mix[:, 1] *= (1 + boost)
    mix[:, 2] *= (1 + 0.3 * boost)
    mix[:, 4] *= (1 - 0.5 * boost)
    mix /= mix.sum(axis=1, keepdims=True)
    u = rng.random(n)[:, None]
    esi = (u > np.cumsum(mix, axis=1)).sum(axis=1) + 1
    esi = np.clip(esi, 1, 5)
    sex = np.where(rng.random(n) < 0.52, "F", "M")
    amb = np.array([rng.random() < AMBULANCE_P[e] for e in esi])
    mode = np.where(amb, "ambulance", np.where(rng.random(n) < 0.04, "other", "walk_in"))
    cc = np.array([rng.choice(COMPLAINTS[e]) for e in esi])
    return age, esi, sex, mode, cc


def simulate(rng, arr, esi):
    n = len(arr)
    consult = np.array([rng.lognormal(np.log(CONSULT_MED[e]), CONSULT_SIG) for e in esi])
    patience = np.array([rng.lognormal(np.log(PATIENCE_MED[e]), PATIENCE_SIG) if e >= 3 else np.inf for e in esi])
    seen = np.full(n, np.nan)
    left = np.full(n, np.nan)  # abandonment time
    waiting = []
    idle = set(range(MAX_DOC))
    events = []  # (time, slot) doctor-free / reactivation events
    boundaries = []
    horizon = arr[-1] + 3 * 1440
    for t in np.arange(0, horizon, 60):
        h = int((t // 60) % 24)
        if h in (0, 7, 15, 23):
            boundaries.append(t)
    bi = 0
    ai = 0

    def eff_prio(i, now):
        return max(esi[i] - (now - arr[i]) / AGING_MIN, AGING_FLOOR if esi[i] >= 3 else esi[i])

    def purge(now):
        keep = []
        for i in waiting:
            if arr[i] + patience[i] <= now:
                left[i] = arr[i] + patience[i]
            else:
                keep.append(i)
        waiting[:] = keep

    def assign(slot, now):
        purge(now)
        if not waiting:
            return False
        best = min(waiting, key=lambda i: (eff_prio(i, now), arr[i]))
        waiting.remove(best)
        seen[best] = now
        heapq.heappush(events, (now + consult[best], slot))
        return True

    while ai < n or waiting or events:
        t_arr = arr[ai] if ai < n else np.inf
        t_ev = events[0][0] if events else np.inf
        t_b = boundaries[bi] if bi < len(boundaries) else np.inf
        nxt = min(t_arr, t_ev, t_b)
        if nxt == np.inf:
            break
        if nxt == t_ev:
            t, slot = heapq.heappop(events)
            if slot_active(slot, t):
                if not assign(slot, t):
                    idle.add(slot)
            else:
                idle.add(slot)
                heapq.heappush(events, (next_active(slot, t), slot)) if waiting else None
        elif nxt == t_arr:
            i = ai
            ai += 1
            waiting.append(i)
            for slot in sorted(idle):
                if slot_active(slot, t_arr):
                    idle.discard(slot)
                    assign(slot, t_arr)
                    break
        else:
            bi += 1
            for slot in sorted(idle):
                if slot_active(slot, t_b) and waiting:
                    idle.discard(slot)
                    assign(slot, t_b)
        if ai >= n and not events and not waiting:
            break
        if ai >= n and not events and waiting and not idle:
            break
    purge(horizon)
    return seen, left, consult


def build(days, seed, scale):
    rng = np.random.default_rng(seed)
    arr = gen_arrivals(rng, days, scale)
    n = len(arr)
    age, esi, sex, mode, cc = draw_patients(rng, n)
    seen, left, consult = simulate(rng, arr, esi)

    disp = np.empty(n, dtype=object)
    dep = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(left[i]) and np.isnan(seen[i]):
            disp[i] = "lwbs"
            dep[i] = left[i]
        elif not np.isnan(seen[i]):
            r = rng.random()
            if r < ADMIT_P[esi[i]]:
                d = "admitted"
            elif r < ADMIT_P[esi[i]] + TRANSFER_P:
                d = "transferred"
            else:
                d = "discharged"
            disp[i] = d
            los = max(rng.lognormal(np.log(LOS_MED[d]), LOS_SIG), consult[i] + 5)
            dep[i] = seen[i] + los
        else:
            disp[i] = "other"  # still waiting when data collection ended
    return arr, age, esi, sex, mode, cc, seen, dep, disp


def to_frame(days, seed, scale, start="2024-01-01", noise=True):
    arr, age, esi, sex, mode, cc, seen, dep, disp = build(days, seed, scale)
    rng = np.random.default_rng(seed + 1)
    t0 = pd.Timestamp(start)
    cutoff = days * 1440.0

    def ts(x):
        return t0 + pd.to_timedelta(x, unit="m")

    df = pd.DataFrame({
        "patient_id": [f"P{i:07d}" for i in range(len(arr))],
        "arrival_time": ts(arr),
        "triage_level": pd.array(esi, dtype="Int64"),
        "seen_time": pd.to_datetime(np.where(np.isnan(seen), pd.NaT, ts(np.nan_to_num(seen)))),
        "departure_time": pd.to_datetime(np.where(np.isnan(dep), pd.NaT, ts(np.nan_to_num(dep)))),
        "disposition": disp,
        "age": age,
        "sex": sex,
        "arrival_mode": mode,
        "chief_complaint": cc,
    })
    # data-collection window closes: later events are unobserved (right-censored)
    end = ts(cutoff)
    df = df[df["arrival_time"] < end].copy()
    df.loc[df["seen_time"] >= end, ["seen_time", "departure_time", "disposition"]] = [pd.NaT, pd.NaT, "other"]
    df.loc[df["departure_time"] >= end, "departure_time"] = pd.NaT
    # round to the minute like real timestamps
    for c in ["arrival_time", "seen_time", "departure_time"]:
        df[c] = df[c].dt.floor("min")
    df["triage_level"] = df["triage_level"].astype("Int64")
    if noise:
        n = len(df)
        df.loc[rng.random(n) < 0.02, "age"] = np.nan
        df.loc[rng.random(n) < 0.03, "arrival_mode"] = np.nan
        df.loc[rng.random(n) < 0.05, "chief_complaint"] = np.nan
        df.loc[rng.random(n) < 0.004, "triage_level"] = pd.NA
        bad = df[df["seen_time"].notna()].sample(frac=0.003, random_state=seed).index
        df.loc[bad, ["arrival_time", "seen_time"]] = df.loc[bad, ["seen_time", "arrival_time"]].values  # swapped timestamps
    return df.reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", type=float, default=9.7)
    ap.add_argument("--out", default="ed_v2.csv")
    ap.add_argument("--no-noise", action="store_true")
    a = ap.parse_args()
    out = to_frame(a.days, a.seed, a.scale, noise=not a.no_noise)
    out.to_csv(a.out, index=False)
    print(f"wrote {len(out)} rows to {a.out}")