#!/usr/bin/env python3
"""
Flask web app for the Ultimate Ops Planning Machine.

Key fairness updates (Oct 2025):
- Removed the Saturday “OIL / 6-day-week” penalty for NSFs.
- NSF availability only checks D (no D+1 exclusion); cover logic belongs to standby, not Ops.
- NSF selection:
  • Bad days (Fri/Sat/3-boarding): balance bad-deal counts vs per-person target first, then totals.
  • Normal days: prefer those over their bad-deal target (preserve under-target for future bad days), then totals.
  • Enforce ≤1 non-cleared per Ops team.
- Leaders (TL/ATL/C2) selection:
  • Bad days: balance bad-deal load inside the role first; then role totals; then overall totals; tiny anti-streak nudge.
  • Normal days: overall totals first; then role totals; tiny anti-streak nudge.
  • ATL-only preferred; fallback to C2+ATL; C2 must not be unavailable on D+1.
"""

import os
from datetime import date, timedelta
from dataclasses import dataclass, field
from typing import List, Set, Tuple, Dict, Optional, Any

from flask import Flask, render_template, request, redirect, url_for, session, flash

# ========= Core helpers =========

def parse_yyyy_mm_dd(s: str) -> date:
    y, m, d = map(int, s.strip().split("-"))
    return date(y, m, d)
def _compact_days_csv(iso_list: List[str]) -> str:
    """
    Turn ISO dates into a compact CSV of DD or DD-DD ranges, sorted ascending.
    Example: ['2025-10-18','2025-10-19','2025-10-20','2025-10-21','2025-10-25','2025-10-31']
      -> '18-21,25,31'
    """
    days: List[int] = []
    for iso in iso_list or []:
        try:
            parts = iso.split("-")
            if len(parts) == 3:
                d = int(parts[2])
                if 1 <= d <= 31:
                    days.append(d)
        except Exception:
            pass
    if not days:
        return ""
    days = sorted(set(days))
    ranges: List[str] = []
    start = prev = days[0]
    for x in days[1:]:
        if x == prev + 1:
            prev = x
            continue
        # close previous run
        ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
        start = prev = x
    # close last run
    ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
    return ",".join(ranges)


def daterange_inclusive(d0: date, d1: date):
    cur = d0
    while cur <= d1:
        yield cur
        cur += timedelta(days=1)

def generate_ops_dates_1_in_3(window_start: date, window_end: date, anchor_ops: date) -> List[date]:
    if window_end < window_start:
        return []
    ops_set: Set[date] = set()
    cur = anchor_ops
    while cur >= window_start:
        if cur <= window_end:
            ops_set.add(cur)
        cur -= timedelta(days=3)
    cur = anchor_ops + timedelta(days=3)
    while cur <= window_end:
        if cur >= window_start:
            ops_set.add(cur)
        cur += timedelta(days=3)
    return sorted(ops_set)

def _expand_dd_list_into_window_dates(line: str, window: Tuple[date, date]) -> Set[date]:
    start, end = window
    if not line or not line.strip():
        return set()
    tokens = [tok.strip() for tok in line.split(",") if tok.strip()]
    days: Set[int] = set()
    for tok in tokens:
        if "-" in tok or ":" in tok:
            try:
                a, b = [int(x) for x in tok.replace(":", "-").split("-", 1)]
                if a > b: a, b = b, a
                for d in range(a, b + 1):
                    if 1 <= d <= 31:
                        days.add(d)
            except Exception:
                continue
        else:
            try:
                d = int(tok)
                if 1 <= d <= 31:
                    days.add(d)
            except Exception:
                continue
    result: Set[date] = set()
    for dt in daterange_inclusive(start, end):
        if dt.day in days:
            result.add(dt)
    return result

def compute_auto_bad_ops(
    all_ops: List[date],
    three_boardings_any: Set[date],
    no_boardings_any: Set[date],
    nstt_cfg_enabled: bool,
    nstt_three: Set[date],
    nstt_no: Set[date],
) -> Set[date]:
    """
    Bad-deal definition:
      - Fridays, Saturdays
      - Any 3-boarding days (global + NSTT lists)
      - EXCEPT 0-boarding Fridays/Saturdays (global + NSTT lists)
    """
    bad: Set[date] = set()
    for d in all_ops:
        if d.weekday() in (4, 5):  # Fri, Sat
            bad.add(d)
    bad |= set(three_boardings_any)
    if nstt_cfg_enabled:
        bad |= set(nstt_three)
    bad -= {d for d in all_ops if d in no_boardings_any and d.weekday() in (4, 5)}
    if nstt_cfg_enabled:
        bad -= {d for d in all_ops if d in nstt_no and d.weekday() in (4, 5)}
    return bad

# ========= Data classes =========

@dataclass
class Person:
    name: str
    unavailable: Set[date]
    unavailable_ops: Set[date]
    cleared: bool = True

@dataclass
class Regular:
    name: str
    roles: Set[str]
    unavailable: Set[date]
    stats_by_role: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: {
            "TL":  {"total": 0, "bad": 0},
            "ATL": {"total": 0, "bad": 0},
            "C2":  {"total": 0, "bad": 0},
        }
    )

@dataclass
class NSTTMember:
    name: str
    types: Set[str] = field(default_factory=set)                 # subset of {'C2','E1'}
    remaining: Dict[str, List[str]] = field(default_factory=dict)  # e.g. {'C2': ['O','E','A']}
    start_date: Optional[date] = None

@dataclass
class NSTTConfig:
    enabled: bool = False
    global_start: Optional[date] = None
    members: Dict[str, NSTTMember] = field(default_factory=dict)
    three_boarding_ops: Set[date] = field(default_factory=set)
    no_boarding_ops: Set[date] = field(default_factory=set)

# ========= Fairness math =========

def compute_bad_targets(people: List[Person], bad_ops: Set[date], team_size: int, carry: Dict[str, int]) -> Dict[str, float]:
    total_bad_slots = len(bad_ops) * team_size
    base = total_bad_slots / max(1, len(people))
    targets: Dict[str, float] = {}
    for p in people:
        adj = base - carry.get(p.name, 0)
        targets[p.name] = max(0.0, adj)
    return targets

def build_leader_bad_targets(regs: List[Regular], bad_ops: Set[date]) -> Dict[str, Dict[str, float]]:
    """
    Compute per-leader per-role target bad-deal counts for the month.
    Simple, stable target: (bad-days for that role) / (#eligible leaders for that role).
    - TL eligible: TL-only
    - ATL eligible: ATL-only + C2,ATL hybrids
    - C2 eligible: anyone with 'C2'
    """
    bad_count = {
        "TL": len(bad_ops),
        "ATL": len(bad_ops),
        "C2": len(bad_ops),
    }

    elig = {
        "TL":  [r for r in regs if r.roles == {"TL"}],
        "ATL": [r for r in regs if r.roles == {"ATL"} or r.roles == {"C2", "ATL"}],
        "C2":  [r for r in regs if "C2" in r.roles],
    }

    targets: Dict[str, Dict[str, float]] = {r.name: {"TL": 0.0, "ATL": 0.0, "C2": 0.0} for r in regs}
    for role in ["TL", "ATL", "C2"]:
        n = max(1, len(elig[role]))
        per = bad_count[role] / n
        for r in elig[role]:
            targets[r.name][role] = per
    return targets

# ========= Leaders planning (improved fairness) =========

def plan_leaders_for_ops(
    all_ops: List[date],
    regs: List[Regular],
    bad_ops: Set[date],
) -> Dict[date, Dict[str, Optional[str]]]:
    """
    Bad day:
      - primary: minimize (role_bad - per-role target bad-days)
      - then:    overall totals (keep whole-pie fair)
      - then:    prefer someone who already carried some role_tot (avoid 'newbie advantage')
      - tiny anti-streak on normal days

    Normal day:
      - overall totals first, then role totals, tiny anti-streak.

    ATL-only preferred; fallback to C2+ATL; C2 must also be free on D+1.
    """
    from copy import deepcopy
    regs = deepcopy(regs)

    # --- Per-role bad-day target (simple equal split among eligible) ---
    def _elig_pool(role: str) -> List[Regular]:
        if role == "TL":
            return [r for r in regs if r.roles == {"TL"}]
        if role == "ATL":
            return [r for r in regs if r.roles == {"ATL"} or r.roles == {"C2", "ATL"}]
        if role == "C2":
            return [r for r in regs if "C2" in r.roles]
        return []

    per_role_target: Dict[str, Dict[str, float]] = {r.name: {"TL": 0.0, "ATL": 0.0, "C2": 0.0} for r in regs}
    num_bad = float(len(bad_ops))
    for role in ("TL", "ATL", "C2"):
        pool = _elig_pool(role)
        if pool:
            tgt = num_bad / len(pool)
            for r in pool:
                per_role_target[r.name][role] = tgt

    def overall_total(r: Regular) -> int:
        s = r.stats_by_role
        return s["TL"]["total"] + s["ATL"]["total"] + s["C2"]["total"]

    last_role_holder: Dict[str, Optional[str]] = {"TL": None, "ATL": None, "C2": None}

    def sort_key(r: Regular, role: str, is_bad: bool) -> tuple:
        s = r.stats_by_role
        role_s = s.get(role, {"total": 0, "bad": 0})
        role_bad = role_s["bad"]
        role_tot = role_s["total"]
        all_tot  = overall_total(r)
        streak   = 1 if (not is_bad and last_role_holder.get(role) == r.name) else 0

        if is_bad:
            # deficit < 0 means behind target => pick sooner
            deficit = role_bad - per_role_target[r.name][role]
            # Put deficit first (most important), then overall totals,
            # then *reverse* role_tot (prefer someone who already shouldered that role),
            # then anti-streak and name.
            return (deficit, all_tot, -role_tot, streak, r.name.lower())
        else:
            return (all_tot, role_tot, streak, r.name.lower())

    out: Dict[date, Dict[str, Optional[str]]] = {}
    for d in sorted(all_ops):
        is_bad = d in bad_ops
        picked: Set[str] = set()
        out[d] = {"TL": None, "ATL": None, "C2": None}

        # TL (TL-only)
        tl_pool = [r for r in regs if r.roles == {"TL"} and d not in r.unavailable and r.name not in picked]
        if tl_pool:
            tl_pool.sort(key=lambda r: sort_key(r, "TL", is_bad))
            tl = tl_pool[0]
            out[d]["TL"] = tl.name
            tl.stats_by_role["TL"]["total"] += 1
            if is_bad: tl.stats_by_role["TL"]["bad"] += 1
            picked.add(tl.name)
            last_role_holder["TL"] = tl.name

        # ATL (ATL-only; else C2+ATL)
        atl_only = [r for r in regs if r.roles == {"ATL"} and d not in r.unavailable and r.name not in picked]
        c2_atl   = [r for r in regs if r.roles == {"C2","ATL"} and d not in r.unavailable and r.name not in picked]
        pool = atl_only if atl_only else c2_atl
        if pool:
            pool.sort(key=lambda r: sort_key(r, "ATL", is_bad))
            atl = pool[0]
            out[d]["ATL"] = atl.name
            atl.stats_by_role["ATL"]["total"] += 1
            if is_bad: atl.stats_by_role["ATL"]["bad"] += 1
            picked.add(atl.name)
            last_role_holder["ATL"] = atl.name

        # C2 (must not be unavailable on D+1)
        c2_pool = [
            r for r in regs
            if ("C2" in r.roles)
            and (d not in r.unavailable)
            and ((d + timedelta(days=1)) not in r.unavailable)
            and (r.name not in picked)
        ]
        if c2_pool:
            c2_pool.sort(key=lambda r: sort_key(r, "C2", is_bad))
            c2 = c2_pool[0]
            out[d]["C2"] = c2.name
            c2.stats_by_role["C2"]["total"] += 1
            if is_bad: c2.stats_by_role["C2"]["bad"] += 1
            picked.add(c2.name)
            last_role_holder["C2"] = c2.name

    return out

def rebalance_leader_bad_days(
    all_ops: List[date],
    bad_ops: Set[date],
    leaders_by_day: Dict[date, Dict[str, Optional[str]]],
    regs: List[Regular],
    leader_bad_targets: Dict[str, Dict[str, float]],
) -> Dict[date, Dict[str, Optional[str]]]:
    """
    Post-planning fairness pass:
    - For each role (C2, ATL, TL), try to move bad-day assignments from leaders above their
      per-role bad-day target to leaders below their target, when a legal swap is possible.
    - Respects role eligibility and C2's D+1 availability rule.
    - Keeps per-day uniqueness (no double-assigning a person to multiple roles the same day).
    """
    regs_by_name: Dict[str, Regular] = {r.name: r for r in regs}

    def _eligible(role: str, r: Regular, d: date, picked_today: Set[str]) -> bool:
        if r.name in picked_today:
            return False
        if d in r.unavailable:
            return False
        if role == "TL" and r.roles != {"TL"}:
            return False
        if role == "ATL":
            if r.roles not in ({"ATL"}, {"C2", "ATL"}):
                return False
        if role == "C2":
            if "C2" not in r.roles:
                return False
            if (d + timedelta(days=1)) in r.unavailable:
                return False
        return True

    def current_counts() -> Dict[str, Dict[str, int]]:
        counts: Dict[str, Dict[str, int]] = {}
        for d in all_ops:
            if d not in leaders_by_day:
                continue
            is_bad = d in bad_ops
            for role, nm in leaders_by_day[d].items():
                if not nm:
                    continue
                counts.setdefault(nm, {}).setdefault(role, 0)
                if is_bad:
                    counts[nm][role] += 1
        return counts

    MAX_SWAPS = 200
    swaps_done = 0

    for role in ["C2", "ATL", "TL"]:
        for _ in range(100):
            changed_this_round = False
            counts = current_counts()

            deficit: List[str] = []
            surplus: List[str] = []
            for r in regs:
                tgt = leader_bad_targets.get(r.name, {}).get(role, 0.0)
                cur = counts.get(r.name, {}).get(role, 0)
                if cur < int(tgt):
                    deficit.append(r.name)
                elif cur > int(tgt + 0.5):
                    surplus.append(r.name)

            if not deficit or not surplus:
                break

            moved_one = False
            for over_nm in surplus:
                for d in sorted(bad_ops):
                    if d not in leaders_by_day:
                        continue
                    if leaders_by_day[d].get(role) != over_nm:
                        continue
                    picked_today = {v for v in leaders_by_day[d].values() if v}
                    for under_nm in deficit:
                        if under_nm == over_nm:
                            continue
                        cand_reg = regs_by_name.get(under_nm)
                        if not cand_reg:
                            continue
                        if not _eligible(role, cand_reg, d, picked_today):
                            continue
                        leaders_by_day[d][role] = under_nm
                        swaps_done += 1
                        changed_this_round = True
                        moved_one = True
                        break
                    if moved_one or swaps_done >= MAX_SWAPS:
                        break
                if moved_one or swaps_done >= MAX_SWAPS:
                    break

            if not changed_this_round or swaps_done >= MAX_SWAPS:
                break

    return leaders_by_day

# ========= NSF planning (improved fairness; no OIL penalty; no D+1 exclusion) =========

def plan_ops_assignments(
    all_ops: List[date],
    bad_ops: Set[date],
    people: List[Person],
    team_size: int,
    target_bad: Dict[str, float],
    nstt_cfg: NSTTConfig,
) -> Tuple[Dict[date, List[str]], Dict[str, Dict[str, int]], Dict[date, Dict[str, str]]]:
    """
    Builds NSF team assignments and (when enabled) assigns NSTT stages per boarding.

    Key NSTT rules implemented:
      • Ops team becomes 5-man when NSTT is enabled.
      • ≤ 2 trainees per Ops day.
      • Sundays = 1 boarding, weekdays = 2 boardings, optional 3-boarding markers, and 0-boarding markers.
      • Per boarding: at most ONE non-observer (E or A).
      • Both trainees may Observe (O) in the same boarding.
      • Same-day chaining is allowed: O→E (same day), or E→A (same day). Never O→E→A in a single day.
      • C2 must clear before E1 (track priority).
      • Urgency priority: fewer future eligible Ops → more remaining stages → next stage (E > A > O).
      • Fairness-first team picking with an urgency nudge (stats/targets drive later balancing).
    """
    # Always 5-man teams when NSTT is on
    if nstt_cfg.enabled:
        team_size = 5

    stats = {p.name: {"total": 0, "bad": 0} for p in people}
    schedule: Dict[date, List[str]] = {}
    nstt_day_roles: Dict[date, Dict[str, str]] = {}

    by_name = {p.name: p for p in people}

    # Mutable copy of per-person remaining NSTT sequences (C2 first, then E1)
    remaining: Dict[str, Dict[str, List[str]]] = {}
    starts: Dict[str, Optional[date]] = {}
    if nstt_cfg.enabled:
        for nm, m in nstt_cfg.members.items():
            remaining[nm] = {
                "C2": list(m.remaining.get("C2", [])),
                "E1": list(m.remaining.get("E1", [])),
            }
            starts[nm] = (m.start_date or nstt_cfg.global_start)

    # --- helpers -------------------------------------------------------------
    def boardings_on(d: date) -> int:
        if not nstt_cfg.enabled:
            return 0
        if d in nstt_cfg.no_boarding_ops:
            return 0
        if d in nstt_cfg.three_boarding_ops:
            return 3
        return 1 if d.weekday() == 6 else 2  # Sun=6 in Python: 0=Mon ... 6=Sun

    def stages_left(nm: str) -> int:
        if nm not in remaining:
            return 0
        return len(remaining[nm].get("C2", [])) + len(remaining[nm].get("E1", []))

    def next_stage(nm: str) -> Optional[Tuple[str, str]]:
        """Strict 'C2 then E1' track priority; returns (track, 'O'/'E'/'A') or None."""
        if nm not in remaining:
            return None
        for track in ("C2", "E1"):
            seq = remaining[nm].get(track, [])
            if seq:
                return (track, seq[0])
        return None

    def future_eligible_ops(nm: str, after_day: date) -> int:
        """Count future Ops days where nm could still train (start date reached, not Ops-constrained)."""
        if nm not in remaining:
            return 0
        p = by_name.get(nm)
        if not p:
            return 0
        st = starts.get(nm)
        cnt = 0
        for dt in all_ops:
            if dt <= after_day:
                continue
            if st and dt < st:
                continue
            if dt in p.unavailable_ops:
                continue
            cnt += 1
        return cnt

    def can_train_today(nm: str, d: date) -> bool:
        """True if nm is an NSTT member with stages left, past start, and not Ops-constrained on d."""
        if nm not in remaining:
            return False
        p = by_name.get(nm)
        if not p:
            return False
        st = starts.get(nm)
        if st and d < st:
            return False
        if d in p.unavailable_ops:
            return False
        return stages_left(nm) > 0

    # --- main daily loop -----------------------------------------------------
    for d in sorted(all_ops):
        is_bad = d in bad_ops

        # ===== Team selection (fairness-first with urgency nudge) =====
        avail = [p for p in people if d not in p.unavailable_ops]
        if not avail:
            schedule[d] = []
            continue

        def team_key(p: Person):
            bad_diff = stats[p.name]["bad"] - target_bad.get(p.name, 0.0)
            total = stats[p.name]["total"]

            # Urgency nudge if this person needs NSTT and can legally train from today onward
            nudge = 0.0
            if nstt_cfg.enabled and p.name in remaining and stages_left(p.name) > 0:
                st = starts.get(p.name)
                if (st is None) or (d >= st):
                    fut = future_eligible_ops(p.name, d)
                    trk_stg = next_stage(p.name)
                    stage = trk_stg[1] if trk_stg else None
                    # prefer action stages, but still give O a small pull so we can chain O→E
                    base = {"E": -1.6, "A": -1.3, "O": -0.9}.get(stage, 0.0)
                    fut_amp = 1.7 if fut <= 2 else (1.4 if fut <= 4 else 1.0)
                    left = stages_left(p.name)
                    left_amp = 1.35 if left >= 3 else (1.15 if left == 2 else 1.0)
                    nudge = base * fut_amp * left_amp

            if is_bad:
                return (bad_diff, total, nudge, p.name.lower())
            else:
                over = 1 if (stats[p.name]["bad"] > target_bad.get(p.name, 0.0) + 1e-4) else 0
                return (-over, total, nudge, p.name.lower())

        ranked = sorted(avail, key=team_key)

        team: List[Person] = []
        noncleared = 0
        for cand in ranked:
            if not cand.cleared and noncleared >= 1:
                continue
            team.append(cand)
            if not cand.cleared:
                noncleared += 1
            if len(team) == team_size:
                break

        # fallback: ensure at least one cleared member is on the team
        if team and all(not t.cleared for t in team):
            extra = next((p for p in ranked if p.cleared and p not in team), None)
            if extra:
                team[-1] = extra

        schedule[d] = [p.name for p in team]

        # fairness stats tick
        for p in team:
            stats[p.name]["total"] += 1
            if is_bad:
                stats[p.name]["bad"] += 1

        # ===== NSTT tagging per day (single pass) =====
        if not nstt_cfg.enabled:
            continue

        nb = boardings_on(d)
        if nb <= 0:
            continue  # normal Ops day with no boardings → no NSTT tags

        team_names = schedule[d]
        eligible = [nm for nm in team_names if can_train_today(nm, d)]
        if not eligible:
            continue

        # Choose up to 2 trainees, most urgent first
        def urgency_key(nm: str):
            fut = future_eligible_ops(nm, d)
            left = stages_left(nm)
            nxt = next_stage(nm)
            # prefer E (0) over A (1) over O (2) for "next stage"
            stage_rank = {"E": 0, "A": 1, "O": 2}
            sr = stage_rank.get(nxt[1], 3) if nxt else 3
            return (fut, -left, sr, nm.lower())

        eligible.sort(key=urgency_key)
        chosen = eligible[:2]
        if not chosen:
            continue

        # Track per-day progress to prevent O+E+A in one day
        did_today: Dict[str, List[str]] = {nm: [] for nm in chosen}
        day_tags: Dict[str, List[str]] = {nm: [] for nm in chosen}

        def pop_if_exact(nm: str, exact_stage: str) -> bool:
            """Advance exactly the trainee's next stage (C2 before E1)."""
            nxt = next_stage(nm)
            if not nxt:
                return False
            track, stg = nxt
            if stg != exact_stage:
                return False
            remaining[nm][track].pop(0)
            did_today[nm].append(exact_stage)
            day_tags[nm].append(exact_stage)
            return True

        def can_take_nonobserver(nm: str) -> bool:
            """A trainee can take E or A now if that is their immediate next stage and they haven't already done 2 stages today."""
            if len(did_today[nm]) >= 2:
                return False
            nxt = next_stage(nm)
            return bool(nxt and nxt[1] in ("E", "A"))

        def can_observe(nm: str) -> bool:
            """A trainee can Observe if 'O' is exactly next, and they've done <2 stages today."""
            if len(did_today[nm]) >= 2:
                return False
            nxt = next_stage(nm)
            return bool(nxt and nxt[1] == "O")

        # Boarding-by-boarding assignment
        for _b in range(nb):
            placed_nonobs = False

            # --- 1) Non-observer slot (E/A) — at most one per boarding ---
            # Prefer chaining within the day: A if someone already did E earlier; else E (ideally after O)
            # Candidate list ordered: (A-ready with prior E today) → (E-ready who already did O today) → (any E-ready) → (any A-ready)
            # Build pools
            a_after_e = [nm for nm in chosen if can_take_nonobserver(nm) and next_stage(nm)[1] == "A" and ("E" in did_today[nm])]
            e_after_o = [nm for nm in chosen if can_take_nonobserver(nm) and next_stage(nm)[1] == "E" and ("O" in did_today[nm])]
            e_plain   = [nm for nm in chosen if can_take_nonobserver(nm) and next_stage(nm)[1] == "E" and ("O" not in did_today[nm])]
            a_plain   = [nm for nm in chosen if can_take_nonobserver(nm) and next_stage(nm)[1] == "A" and ("E" not in did_today[nm])]

            picked_nm = None
            for pool in (a_after_e, e_after_o, e_plain, a_plain):
                if pool:
                    # within a pool, prefer the more urgent trainee
                    pool.sort(key=urgency_key)
                    picked_nm = pool[0]
                    break

            if picked_nm:
                pop_if_exact(picked_nm, next_stage(picked_nm)[1])
                placed_nonobs = True

            # --- 2) Observers — both trainees may Observe this boarding if it's exactly next ---
            for nm in chosen:
                if can_observe(nm):
                    # Permit observing even if another trainee just used the non-observer slot
                    # (still ≤1 E/A this boarding).
                    # Also allow observe→execute chaining later in the same day.
                    pop_if_exact(nm, "O")

            # Safety: ensure no one exceeds 2 stages in the day
            for nm in chosen:
                if len(did_today[nm]) > 2:
                    # This should never happen by construction; guard anyway.
                    did_today[nm] = did_today[nm][:2]

        # Finalize tags: map to "O", "E", "A", "O&E", "E&A"
        tags = {nm: "&".join(v) for nm, v in day_tags.items() if v}
        if tags:
            nstt_day_roles[d] = tags

    return schedule, stats, nstt_day_roles


def replan_nstt_tags(
    all_ops: List[date],
    nsf_schedule: Dict[date, List[str]],
    people: List[Person],
    nstt_cfg: NSTTConfig,
) -> Dict[date, Dict[str, str]]:
    """
    Deprecated: NSTT tagging is finalized during plan_ops_assignments in a single pass.
    Keeping this as a no-op for backward compatibility with templates.
    """
    return {}





# ========= Standby (unchanged logic from your improved version) =========

def plan_standby_with_leaders(
    all_ops: List[date],
    nsf_schedule: Dict[date, List[str]],
    people: List[Person],
    leaders_by_day: Dict[date, Dict[str, Optional[str]]],
    regs: List[Regular],
    window_start: date,
    window_end: date,
) -> Dict[date, List[str]]:
    regs_by_name = {r.name: r for r in regs}
    by_name = {p.name: p for p in people}
    standby: Dict[date, List[str]] = {}

    def sub_for_role(role: str, sday: date, picked: Set[str]) -> Optional[str]:
        if role == "ATL":
            atl_only = [r for r in regs if r.roles == {"ATL"} and sday not in r.unavailable and r.name not in picked]
            if atl_only:
                atl_only.sort(key=lambda r: (r.stats_by_role["ATL"]["total"], r.name.lower()))
                return atl_only[0].name
            c2_atl = [r for r in regs if r.roles == {"C2","ATL"} and sday not in r.unavailable and r.name not in picked]
            if c2_atl:
                c2_atl.sort(key=lambda r: (r.stats_by_role["ATL"]["total"], r.name.lower()))
                return c2_atl[0].name
            return None
        else:
            pool = [r for r in regs if role in r.roles and sday not in r.unavailable and r.name not in picked]
            if not pool:
                return None
            pool.sort(key=lambda r: (r.stats_by_role[role]["total"], r.name.lower()))
            return pool[0].name

    for d in sorted(all_ops):
        sday = d - timedelta(days=1)
        if sday < window_start or sday > window_end:
            continue

        picks: List[str] = []
        picked_set: Set[str] = set()

        today = leaders_by_day.get(d, {"TL": None, "ATL": None, "C2": None})
        for role in ("TL","ATL","C2"):
            nm = today.get(role)
            if nm:
                r = regs_by_name.get(nm)
                if r and sday not in r.unavailable:
                    picks.append(nm); picked_set.add(nm)
                else:
                    sub = sub_for_role(role, sday, picked_set)
                    picks.append(f"({sub})" if sub else "")
                    if sub: picked_set.add(sub)
            else:
                sub = sub_for_role(role, sday, picked_set)
                picks.append(sub or "")
                if sub: picked_set.add(sub)

        # NSFs (3 pax; ≤1 non-cleared). Prefer next-day NSFs if free on standby day.
        want_nsf = 3
        nsf_picks: List[str] = []
        noncleared_cnt = 0
        blocked_due_to_constraint = 0
        next_day = nsf_schedule.get(d, [])

        for nm in next_day:
            if len(nsf_picks) >= want_nsf: break
            p = by_name.get(nm)
            if not p: continue
            if sday in p.unavailable:
                blocked_due_to_constraint += 1
                continue
            if (not p.cleared) and noncleared_cnt >= 1:
                continue
            nsf_picks.append(nm)
            if not p.cleared: noncleared_cnt += 1

        if len(nsf_picks) < want_nsf:
            team_set = set(nsf_picks)
            for p in people:
                if len(nsf_picks) >= want_nsf: break
                if p.name in team_set: continue
                if sday in p.unavailable: continue
                if p.name in next_day and (not p.cleared) and noncleared_cnt >= 1:
                    continue
                if (not p.cleared) and noncleared_cnt >= 1:
                    continue
                if blocked_due_to_constraint > 0:
                    nsf_picks.append(f"({p.name})")
                    blocked_due_to_constraint -= 1
                else:
                    nsf_picks.append(p.name)
                if not p.cleared: noncleared_cnt += 1

        standby[sday] = picks + nsf_picks

    return standby

# ========= Tables for templates =========

def build_tables(
    all_ops: List[date],
    bad_ops: Set[date],
    leaders_by_day: Dict[date, Dict[str, Optional[str]]],
    nsf_schedule: Dict[date, List[str]],
    nstt_roles: Dict[date, Dict[str, str]],
    stats: Dict[str, Dict[str, int]],
    carry: Dict[str, int],
    standby: Dict[date, List[str]],
    regs: List[Regular],
) -> Dict[str, Any]:
    # Pretty ops rows (unchanged)
    ops_rows: List[List[str]] = []
    # Flat ops rows for copy-to-Excel: [Date, TL, ATL, C2, NSF1..NSF5, NSTT1, NSTT2]
    ops_rows_flat: List[List[str]] = []
    ops_headers_flat = ["Date", "TL", "ATL", "C2", "NSF1", "NSF2", "NSF3", "NSF4", "NSF5", "NSTT1", "NSTT2"]

    for d in sorted(all_ops):
        day_str = f"{d.day:02d} {d.strftime('%b').lower()} ({d.strftime('%a').lower()})"
        if d in bad_ops:
            day_str = f"*{day_str}*"

        leaders = leaders_by_day.get(d, {"TL": None, "ATL": None, "C2": None})
        nsf_names = nsf_schedule.get(d, [])
        nstt_tags = nstt_roles.get(d, {})

        # Pretty string (unchanged)
        normal_nsfs = [nm for nm in nsf_names if nm not in nstt_tags]
        tagged_nsfs = [f"{nm}({nstt_tags[nm]})" for nm in nsf_names if nm in nstt_tags]
        nsf_str = ", ".join(normal_nsfs + tagged_nsfs) if nsf_names else "—"
        ops_rows.append([
            day_str,
            leaders.get("TL") or "—",
            leaders.get("ATL") or "—",
            leaders.get("C2") or "—",
            nsf_str,
        ])

        # Flat row
        nsf_slots = (nsf_names + ["", "", "", "", ""])[:5]
        nstt_list = [f"{nm}:{nstt_tags[nm]}" for nm in nsf_names if nm in nstt_tags]
        nstt_slots = (nstt_list + ["", ""])[:2]
        ops_rows_flat.append([
            f"{d.strftime('%Y-%m-%d')}",
            leaders.get("TL") or "",
            leaders.get("ATL") or "",
            leaders.get("C2") or "",
            *nsf_slots,
            *nstt_slots
        ])

    # Standby pretty rows (unchanged)
    standby_rows: List[List[str]] = []
    # Flat standby rows: [Date, TL, ATL, C2, NSF1, NSF2, NSF3]
    standby_rows_flat: List[List[str]] = []
    standby_headers_flat = ["Date", "TL", "ATL", "C2", "NSF1", "NSF2", "NSF3"]

    for sday in sorted(standby.keys()):
        day_str = f"{sday.day:02d} {sday.strftime('%b').lower()} ({sday.strftime('%a').lower()})"
        team = standby[sday]
        tl = team[0] if len(team) > 0 else "—"
        atl = team[1] if len(team) > 1 else "—"
        c2 = team[2] if len(team) > 2 else "—"
        nsfs = team[3:] if len(team) > 3 else []
        nsf_str = ", ".join(nsfs) if nsfs else "—"
        standby_rows.append([day_str, tl or "—", atl or "—", c2 or "—", nsf_str])

        nsf_slots = (nsfs + ["", "", ""])[:3]
        standby_rows_flat.append([
            f"{sday.strftime('%Y-%m-%d')}", tl or "", atl or "", c2 or "", *nsf_slots
        ])

    total_ops_count = len(all_ops)
    nsf_summary_rows: List[List[str]] = []
    for name in sorted(stats.keys(), key=str.lower):
        total = stats[name]["total"]
        bad = stats[name]["bad"]
        co = carry.get(name, 0)
        nsf_summary_rows.append([name, f"{total}/{total_ops_count}", str(bad), str(co)])

    leader_summary_rows: List[List[str]] = []
    role_order = ["TL", "ATL", "C2"]
    leader_stats: Dict[str, Dict[str, Dict[str, int]]] = {}
    for d in sorted(all_ops):
        is_bad = d in bad_ops
        leaders = leaders_by_day.get(d, {})
        for role in role_order:
            nm = leaders.get(role)
            if not nm:
                continue
            leader_stats.setdefault(nm, {}).setdefault(role, {"total": 0, "bad": 0})
            leader_stats[nm][role]["total"] += 1
            if is_bad:
                leader_stats[nm][role]["bad"] += 1
    for role in role_order:
        role_rows: List[List[str]] = []
        for name, roles in leader_stats.items():
            dct = roles.get(role)
            if not dct:
                continue
            role_rows.append([name, role, f"{dct['total']}/{total_ops_count}", str(dct["bad"])])
        role_rows.sort(key=lambda x: x[0].lower())
        if role_rows:
            if leader_summary_rows:
                leader_summary_rows.append(["", "", "", ""])
            leader_summary_rows.extend(role_rows)

    return {
        "ops_rows": ops_rows,
        "ops_rows_flat": ops_rows_flat,
        "ops_headers_flat": ops_headers_flat,
        "standby_rows": standby_rows,
        "standby_rows_flat": standby_rows_flat,
        "standby_headers_flat": standby_headers_flat,
        "nsf_summary_rows": nsf_summary_rows,
        "leader_summary_rows": leader_summary_rows,
    }

# ========= Flask app + routes =========

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "dev_secret_key_change_me")

def reset_session():
    for k in [
        "start_date","end_date","anchor_date",
        "num_nsf","num_leaders","nsf_team_size",
        "three_boardings_any","no_boardings_any",
        "nstt_enabled","roster","leaders","nstt_cfg","carryover",
    ]:
        session.pop(k, None)

@app.route("/")
def index():
    return render_template(
        "index.html",
        start_date=session.get("start_date", ""),
        end_date=session.get("end_date", ""),
        anchor_date=session.get("anchor_date", ""),
        num_nsf=session.get("num_nsf", ""),
        num_leaders=session.get("num_leaders", ""),
        nstt_enabled=session.get("nstt_enabled", False),
    )

@app.route("/step1", methods=["POST"])
def step1():
    try:
        start_date = parse_yyyy_mm_dd(request.form["start_date"])
        end_date   = parse_yyyy_mm_dd(request.form["end_date"])
        anchor     = parse_yyyy_mm_dd(request.form["anchor_date"])
    except Exception:
        flash("Please enter valid dates in YYYY-MM-DD format.")
        return redirect(url_for("index"))
    if end_date < start_date:
        flash("End date must not be before start date.")
        return redirect(url_for("index"))

    session["start_date"] = start_date.isoformat()
    session["end_date"]   = end_date.isoformat()
    session["anchor_date"] = anchor.isoformat()

    try:
        num_nsf = int(request.form["num_nsf"])
        num_leaders = int(request.form["num_leaders"])
    except Exception:
        flash("Please provide numeric counts for NSFs and leaders.")
        return redirect(url_for("index"))
    if num_nsf <= 0 or num_leaders < 0:
        flash("NSFs must be positive and leaders non-negative.")
        return redirect(url_for("index"))

    session["num_nsf"] = num_nsf
    session["num_leaders"] = num_leaders
    session["nstt_enabled"] = "nstt_enabled" in request.form
    return redirect(url_for("nsfs"))

@app.route("/nsfs", methods=["GET","POST"])
def nsfs():
    if request.method == "POST":
        num = session.get("num_nsf")
        if not isinstance(num, int) or num <= 0:
            flash("Session expired or invalid NSF count. Please start again.")
            return redirect(url_for("index"))
        try:
            start_date = parse_yyyy_mm_dd(session["start_date"])
            end_date   = parse_yyyy_mm_dd(session["end_date"])
        except Exception:
            flash("Session dates missing. Please start again.")
            return redirect(url_for("index"))
        roster: List[Dict[str, Any]] = []
        start_date = parse_yyyy_mm_dd(session["start_date"])
        end_date   = parse_yyyy_mm_dd(session["end_date"])
        for i in range(1, num + 1):
            name = request.form.get(f"nsf_name_{i}", "").strip()
            unavailable_str = request.form.get(f"nsf_unavail_{i}", "").strip()
            cleared = request.form.get(f"nsf_cleared_{i}") == "on"
            if not name:
                flash("All NSFs must have a name.")
                return redirect(url_for("nsfs"))
            unavail = _expand_dd_list_into_window_dates(unavailable_str, (start_date, end_date))
            roster.append({
                "name": name,
                "unavailable": [d.isoformat() for d in unavail],
                "cleared": cleared,
            })
        session["roster"] = roster
        try:
            team_size = int(request.form.get("team_size", "3"))
        except Exception:
            team_size = 3
        if team_size < 1 or team_size > 5:
            team_size = 3
        session["nsf_team_size"] = team_size
        return redirect(url_for("leaders"))

        # --- NSFs GET branch ---
    num_nsf = session.get("num_nsf")
    if not num_nsf:
        return redirect(url_for("index"))

    raw_roster = session.get("roster", [])
    existing_roster = []
    for r in raw_roster:
        existing_roster.append({
            "name": r.get("name", ""),
            "unavail_csv": _compact_days_csv(r.get("unavailable", [])),
            "cleared": bool(r.get("cleared", False)),
        })

    return render_template(
        "nsfs.html",
        num_nsf=num_nsf,
        existing_roster=existing_roster,
        team_size=session.get("nsf_team_size", 3),
    )

    

@app.route("/leaders", methods=["GET","POST"])
def leaders():
    if request.method == "POST":
        num_leaders = session.get("num_leaders")
        start_date = parse_yyyy_mm_dd(session["start_date"])
        end_date   = parse_yyyy_mm_dd(session["end_date"])
        regs: List[Dict[str, Any]] = []
        for i in range(1, num_leaders + 1):
            name = request.form.get(f"leader_name_{i}", "").strip()
            roles_raw = request.form.get(f"leader_roles_{i}", "").strip().upper().replace(" ", "")
            unavailable_str = request.form.get(f"leader_unavail_{i}", "").strip()
            if not name:
                flash("All leaders must have a name.")
                return redirect(url_for("leaders"))
            roles = set([tok for tok in roles_raw.split(",") if tok])
            if roles not in ({"TL"}, {"ATL"}, {"C2"}, {"C2","ATL"}):
                flash("Leader roles must be TL; ATL; C2; or C2,ATL.")
                return redirect(url_for("leaders"))
            unavail = _expand_dd_list_into_window_dates(unavailable_str, (start_date, end_date))
            regs.append({
                "name": name,
                "roles": list(roles),
                "unavailable": [d.isoformat() for d in unavail],
            })
        session["leaders"] = regs

        three_any_raw = request.form.get("three_boardings_any", "").strip()
        no_any_raw    = request.form.get("no_boardings_any", "").strip()
        three_dates = _expand_dd_list_into_window_dates(three_any_raw, (start_date, end_date))
        no_dates    = _expand_dd_list_into_window_dates(no_any_raw, (start_date, end_date))
        session["three_boardings_any"] = [d.isoformat() for d in (three_dates or [])]
        session["no_boardings_any"]    = [d.isoformat() for d in (no_dates or [])]

        if session.get("nstt_enabled"):
            return redirect(url_for("nstt"))
        else:
            return redirect(url_for("carry"))

        # --- Leaders GET branch ---
    num_leaders = session.get("num_leaders")
    if num_leaders is None:
        return redirect(url_for("index"))

    raw_leaders = session.get("leaders", [])

    def roles_to_code(roles_list):
        s = set(roles_list or [])
        if s == {"TL"}: return "TL"
        if s == {"ATL"}: return "ATL"
        if s == {"C2"}: return "C2"
        if s == {"C2", "ATL"}: return "C2,ATL"
        return ""

    existing_leaders = []
    for r in raw_leaders:
        existing_leaders.append({
            "name": r.get("name", ""),
            "roles_code": roles_to_code(r.get("roles")),
            "unavail_csv": _compact_days_csv(r.get("unavailable", [])),
        })

    existing_three_any = _compact_days_csv(session.get("three_boardings_any", []))
    existing_no_any = _compact_days_csv(session.get("no_boardings_any", []))

    return render_template(
        "leaders.html",
        num_leaders=num_leaders,
        existing_leaders=existing_leaders,
        existing_three_any=existing_three_any,
        existing_no_any=existing_no_any,
    )

    

@app.route("/nstt", methods=["GET","POST"])
def nstt():
    if not session.get("nstt_enabled"):
        return redirect(url_for("carry"))

    if request.method == "POST":
        start_date = parse_yyyy_mm_dd(session["start_date"])
        end_date   = parse_yyyy_mm_dd(session["end_date"])

        gl_start_raw = request.form.get("nstt_global_start", "").strip()
        gl_start = parse_yyyy_mm_dd(gl_start_raw) if gl_start_raw else None

        three_raw = request.form.get("nstt_three_boardings", "").strip()
        no_raw    = request.form.get("nstt_no_boardings", "").strip()
        three_dates = _expand_dd_list_into_window_dates(three_raw, (start_date, end_date))
        no_dates    = _expand_dd_list_into_window_dates(no_raw, (start_date, end_date))

        nstt_data: Dict[str, Any] = {
            "enabled": True,
            "global_start": gl_start.isoformat() if gl_start else None,
            "three_boardings": [d.isoformat() for d in three_dates],
            "no_boardings":    [d.isoformat() for d in no_dates],
            "members": [],
        }

        try:
            num_members = int(request.form.get("nstt_num_members", "0"))
        except Exception:
            num_members = 0

        roster = session.get("roster", [])
        name_map = {p["name"].lower(): p["name"] for p in roster}

        for i in range(1, num_members + 1):
            raw_name = request.form.get(f"nstt_name_{i}", "").strip()
            if raw_name.lower() not in name_map:
                flash("NSTT participant names must match NSF roster names.")
                return redirect(url_for("nstt"))
            nm = name_map[raw_name.lower()]

            types_raw = request.form.get(f"nstt_types_{i}", "").strip().upper().replace(" ", "").replace("+", ",")
            toks = [t for t in types_raw.split(",") if t]
            types = set(x for x in toks if x in {"C2","E1"})

            rem: Dict[str, List[str]] = {}
            if "C2" in types:
                c2_rem_raw = request.form.get(f"nstt_c2_rem_{i}", "").strip().upper()
                if c2_rem_raw:
                    c2_toks = [t.strip() for t in c2_rem_raw.split(",") if t.strip()]
                    order = [x for x in ["O","E","A"] if x in c2_toks]
                    if "O" in order and "E" in order and "A" not in order:
                        order.append("A")
                    rem["C2"] = order
            if "E1" in types:
                e1_rem_raw = request.form.get(f"nstt_e1_rem_{i}", "").strip().upper()
                if e1_rem_raw:
                    e1_toks = [t.strip() for t in e1_rem_raw.split(",") if t.strip()]
                    order = [x for x in ["O","E","A"] if x in e1_toks]
                    if "O" in order and "E" in order and "A" not in order:
                        order.append("A")
                    rem["E1"] = order

            pstart_raw = request.form.get(f"nstt_start_{i}", "").strip()
            pstart = parse_yyyy_mm_dd(pstart_raw) if pstart_raw else None

            nstt_data["members"].append({
                "name": nm,
                "types": list(types),
                "remaining": rem,
                "start": pstart.isoformat() if pstart else None,
            })

        session["nstt_cfg"] = nstt_data
        return redirect(url_for("carry"))

        # -------- GET branch (safe prefill) --------
    cfg = session.get("nstt_cfg") or {}

    def days_csv(iso_list):
        out = []
        for iso in (iso_list or []):
            try:
                parts = iso.split("-")
                if len(parts) == 3:
                    out.append(parts[2])
            except Exception:
                pass
        return ",".join(out)

    existing_global_start = cfg.get("global_start") or ""
    existing_three_boardings = days_csv(cfg.get("three_boardings"))
    existing_no_boardings    = days_csv(cfg.get("no_boardings"))

    existing_members = []
    for m in cfg.get("members", []):
        existing_members.append({
            "name": m.get("name",""),
            "types": ",".join(m.get("types", [])),
            "c2_rem": ",".join(m.get("remaining", {}).get("C2", [])),
            "e1_rem": ",".join(m.get("remaining", {}).get("E1", [])),
            "start": m.get("start","") or "",
        })

    # How many rows to show by default
    default_rows = max(len(existing_members), 0)
    return render_template(
        "nstt.html",
        existing_global_start=existing_global_start,
        existing_three_boardings=existing_three_boardings,
        existing_no_boardings=existing_no_boardings,
        existing_members=existing_members,
        default_member_rows=default_rows,
    )



@app.route("/carry", methods=["GET","POST"])
def carry():
    if request.method == "POST":
        s = request.form.get("carryover", "").strip()
        roster = session.get("roster", [])
        carry: Dict[str, int] = {p["name"]: 0 for p in roster}
        if s:
            tokens = [tok.strip() for tok in s.split(",") if tok.strip()]
            for tok in tokens:
                if ":" not in tok:
                    continue
                name, val = [x.strip() for x in tok.split(":", 1)]
                for p in roster:
                    if p["name"].lower() == name.lower():
                        try:
                            carry[p["name"]] = int(val)
                        except Exception:
                            pass
        session["carryover"] = carry
        return redirect(url_for("results"))
    return render_template("carry.html")

@app.route("/results")
def results():
    required = ["start_date","end_date","anchor_date","roster","leaders"]
    missing = [k for k in required if k not in session]
    if missing:
        flash("Setup incomplete. Please start again.")
        return redirect(url_for("index"))

    start_date = parse_yyyy_mm_dd(session["start_date"])
    end_date   = parse_yyyy_mm_dd(session["end_date"])
    anchor     = parse_yyyy_mm_dd(session["anchor_date"])
    all_ops = generate_ops_dates_1_in_3(start_date, end_date, anchor)

    # NSFs
    roster_data = session.get("roster", [])
    people: List[Person] = []
    all_ops_set = set(all_ops)
    for p in roster_data:
        name = p["name"]
        unavail = set(parse_yyyy_mm_dd(x) for x in p.get("unavailable", []))
        unavail_ops = unavail & all_ops_set
        cleared = p.get("cleared", False)
        people.append(Person(name=name, unavailable=unavail, unavailable_ops=unavail_ops, cleared=cleared))
    team_size = session.get("nsf_team_size", 3)

    # Leaders
    regs_data = session.get("leaders", [])
    regs: List[Regular] = []
    for r in regs_data:
        name = r["name"]
        roles = set(r.get("roles", []))
        unavail = set(parse_yyyy_mm_dd(x) for x in r.get("unavailable", []))
        regs.append(Regular(name=name, roles=roles, unavailable=unavail))

    # Global boarding markers
    three_any = set(parse_yyyy_mm_dd(x) for x in session.get("three_boardings_any", []))
    no_any    = set(parse_yyyy_mm_dd(x) for x in session.get("no_boardings_any", []))

    # NSTT
    nstt_cfg_data = session.get("nstt_cfg")
    if nstt_cfg_data:
        nstt_cfg = NSTTConfig(enabled=True)
        gs = nstt_cfg_data.get("global_start")
        nstt_cfg.global_start = parse_yyyy_mm_dd(gs) if gs else None
        nstt_cfg.three_boarding_ops = set(parse_yyyy_mm_dd(x) for x in nstt_cfg_data.get("three_boardings", []))
        nstt_cfg.no_boarding_ops    = set(parse_yyyy_mm_dd(x) for x in nstt_cfg_data.get("no_boardings", []))
        for m in nstt_cfg_data.get("members", []):
            nm = m["name"]
            types = set(m.get("types", []))
            remaining: Dict[str, List[str]] = {k: list(v) for k, v in m.get("remaining", {}).items()}
            pstart = m.get("start")
            st = parse_yyyy_mm_dd(pstart) if pstart else None
            nstt_cfg.members[nm] = NSTTMember(name=nm, types=types, remaining=remaining, start_date=st)
    else:
        nstt_cfg = NSTTConfig(enabled=False)

    # Bad ops, carry, targets
    bad_ops = compute_auto_bad_ops(all_ops, three_any, no_any, nstt_cfg.enabled, nstt_cfg.three_boarding_ops, nstt_cfg.no_boarding_ops)
    carry   = session.get("carryover", {})
    target_bad = compute_bad_targets(people, bad_ops, team_size if not nstt_cfg.enabled else 5, carry)

    # Plan
    nsf_schedule, stats, nstt_roles = plan_ops_assignments(all_ops, bad_ops, people, team_size, target_bad, nstt_cfg)



    # Leaders: initial plan
    leader_bad_targets = build_leader_bad_targets(regs, bad_ops)
    leaders_by_day = plan_leaders_for_ops(all_ops, regs, bad_ops)

    # Leaders: fairness audit & repair (shift bad days from over-target to under-target where legal)
    leaders_by_day = rebalance_leader_bad_days(all_ops, bad_ops, leaders_by_day, regs, leader_bad_targets)

    # Standby after final leader plan
    standby = plan_standby_with_leaders(all_ops, nsf_schedule, people, leaders_by_day, regs, start_date, end_date)

    tables = build_tables(all_ops, bad_ops, leaders_by_day, nsf_schedule, nstt_roles, stats, carry, standby, regs)

    return render_template(
        "results.html",
        ops_rows=tables["ops_rows"],
        standby_rows=tables["standby_rows"],
        nsf_summary_rows=tables["nsf_summary_rows"],
        leader_summary_rows=tables["leader_summary_rows"],
    )

@app.route("/reset")
def reset():
    reset_session()
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
