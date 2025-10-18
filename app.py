#!/usr/bin/env python3
"""
Flask application that exposes the Ultimate Ops Planning Machine as a web app.

The goal of this application is to take the interactive, command‑line based
scheduler from the provided script and turn it into a series of web forms. Once
the user provides all necessary inputs (date ranges, rosters, constraints,
leaders, NSTT configuration and carry‑over), the app computes the Ops and
standby schedules and displays the results in nicely formatted tables.

To keep the code maintainable we reuse the existing scheduling functions from
the original script. Where needed, some helper functions have been refactored
to accept values from HTTP forms instead of reading from stdin.

Run the application with ``python app.py`` and visit the printed URL in your
browser. Each page in the workflow stores its data in the Flask session so
multi‑step forms can be navigated safely. A secret key is required to use the
session; for demonstration purposes a static key is provided below. In a
production environment you should set ``FLASK_SECRET`` in your environment.
"""

import os
from datetime import date, timedelta
from dataclasses import dataclass, field
from typing import List, Set, Tuple, Dict, Optional

from flask import Flask, render_template, request, redirect, url_for, session, flash

# ====== Helpers copied from the original script ======

def parse_yyyy_mm_dd(s: str) -> date:
    """Parse a YYYY‑MM‑DD string into a date object."""
    y, m, d = map(int, s.strip().split("-"))
    return date(y, m, d)


def daterange_inclusive(d0: date, d1: date):
    """Yield each date between d0 and d1 inclusive."""
    cur = d0
    while cur <= d1:
        yield cur
        cur += timedelta(days=1)


def generate_ops_dates_1_in_3(window_start: date, window_end: date, anchor_ops: date) -> List[date]:
    """
    Generate Ops dates spaced every three days around the anchor. Returns a list
    sorted chronologically.
    """
    if window_end < window_start:
        return []
    ops_set: Set[date] = set()
    # backwards from anchor
    cur = anchor_ops
    while cur >= window_start:
        if cur <= window_end:
            ops_set.add(cur)
        cur -= timedelta(days=3)
    # forwards from anchor
    cur = anchor_ops + timedelta(days=3)
    while cur <= window_end:
        if cur >= window_start:
            ops_set.add(cur)
        cur += timedelta(days=3)
    return sorted(ops_set)


def _expand_dd_list_into_window_dates(line: str, window: Tuple[date, date]) -> Set[date]:
    """
    Expand a comma separated list of days or ranges into concrete dates within a
    given window. For example, ``"5, 10-12"`` returns all dates in the window
    whose day‑of‑month is 5, 10, 11 or 12.
    """
    start, end = window
    if not line.strip():
        return set()
    tokens = [tok.strip() for tok in line.split(",") if tok.strip()]
    days: Set[int] = set()
    for tok in tokens:
        if "-" in tok or ":" in tok:
            try:
                a, b = [int(x) for x in tok.replace(":", "-").split("-", 1)]
                if a > b:
                    a, b = b, a
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


def compute_auto_bad_ops(all_ops: List[date], three_boardings_any: Set[date], no_boardings_any: Set[date], nstt_cfg_enabled: bool, nstt_three: Set[date], nstt_no: Set[date]) -> Set[date]:
    """
    Compute the set of bad‑deal Ops dates. Bad deals include Fridays, Saturdays
    and any 3‑boarding days, except that 0‑boarding days on Fri/Sat are not bad.
    When NSTT is enabled its own 3‑/0‑boardings lists are also considered.
    """
    bad: Set[date] = set()
    # Fridays (weekday() == 4) and Saturdays (weekday() == 5)
    for d in all_ops:
        if d.weekday() == 4 or d.weekday() == 5:
            bad.add(d)
    # Always count 3‑boardings as bad
    bad |= set(three_boardings_any)
    if nstt_cfg_enabled:
        bad |= set(nstt_three)
    # Remove 0‑boardings on Fri/Sat
    bad -= {d for d in all_ops if d in no_boardings_any and d.weekday() in (4, 5)}
    if nstt_cfg_enabled:
        bad -= {d for d in all_ops if d in nstt_no and d.weekday() in (4, 5)}
    return bad


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
            "TL": {"total": 0, "bad": 0},
            "ATL": {"total": 0, "bad": 0},
            "C2": {"total": 0, "bad": 0},
        }
    )


@dataclass
class NSTTMember:
    name: str
    types: Set[str] = field(default_factory=set)
    remaining: Dict[str, List[str]] = field(default_factory=dict)
    start_date: Optional[date] = None


@dataclass
class NSTTConfig:
    enabled: bool = False
    global_start: Optional[date] = None
    members: Dict[str, NSTTMember] = field(default_factory=dict)
    three_boarding_ops: Set[date] = field(default_factory=set)
    no_boarding_ops: Set[date] = field(default_factory=set)


def compute_bad_targets(people: List[Person], bad_ops: Set[date], team_size: int, carry: Dict[str, int]) -> Dict[str, float]:
    """
    Compute target number of bad‑deal assignments per NSF. Each person should
    contribute fairly to the total number of bad slots, adjusted by carryover.
    """
    total_bad_slots = len(bad_ops) * team_size
    base = total_bad_slots / max(1, len(people))
    targets: Dict[str, float] = {}
    for p in people:
        adj = base - carry.get(p.name, 0)
        if adj < 0:
            adj = 0
        targets[p.name] = adj
    return targets


def plan_leaders_for_ops(all_ops: List[date], regs: List[Regular], bad_ops: Set[date]) -> Dict[date, Dict[str, Optional[str]]]:
    """Plan leader assignments for each Ops day. Reuses fairness logic from the script."""
    from copy import deepcopy
    # We'll copy regs so that stats don't persist across sessions; each call fresh
    regs = deepcopy(regs)
    def _leader_overall_total(r: Regular) -> int:
        st = r.stats_by_role
        return st["TL"]["total"] + st["ATL"]["total"] + st["C2"]["total"]

    def _reg_key_for_role_balanced(r: Regular, role: str, is_bad: bool):
        overall_tot = _leader_overall_total(r)
        role_st = r.stats_by_role.get(role, {"total": 0, "bad": 0})
        role_bad = role_st["bad"]
        role_tot = role_st["total"]
        return (
            overall_tot,
            (role_bad if is_bad else 0),
            role_tot,
            r.name.lower(),
        )

    by_day: Dict[date, Dict[str, Optional[str]]] = {}
    for d in sorted(all_ops):
        is_bad = d in bad_ops
        picked: Set[str] = set()
        by_day[d] = {"TL": None, "ATL": None, "C2": None}
        # TL: TL only
        tl_pool = [r for r in regs if r.roles == {"TL"} and d not in r.unavailable and r.name not in picked]
        if tl_pool:
            tl_pool.sort(key=lambda r: _reg_key_for_role_balanced(r, "TL", is_bad))
            tl = tl_pool[0]
            by_day[d]["TL"] = tl.name
            tl.stats_by_role["TL"]["total"] += 1
            if is_bad:
                tl.stats_by_role["TL"]["bad"] += 1
            picked.add(tl.name)
        # ATL: prefer ATL only, fallback C2+ATL
        atl_only = [r for r in regs if r.roles == {"ATL"} and d not in r.unavailable and r.name not in picked]
        c2_atl = [r for r in regs if r.roles == {"C2", "ATL"} and d not in r.unavailable and r.name not in picked]
        atl: Optional[Regular] = None
        if atl_only:
            atl_only.sort(key=lambda r: _reg_key_for_role_balanced(r, "ATL", is_bad))
            atl = atl_only[0]
        elif c2_atl:
            c2_atl.sort(key=lambda r: _reg_key_for_role_balanced(r, "ATL", is_bad))
            atl = c2_atl[0]
        if atl:
            by_day[d]["ATL"] = atl.name
            atl.stats_by_role["ATL"]["total"] += 1
            if is_bad:
                atl.stats_by_role["ATL"]["bad"] += 1
            picked.add(atl.name)
        # C2: any with C2, not already picked, and also not unavailable next day
        c2_pool = [
            r
            for r in regs
            if ("C2" in r.roles)
            and (d not in r.unavailable)
            and ((d + timedelta(days=1)) not in r.unavailable)
            and (r.name not in picked)
        ]
        if c2_pool:
            c2_pool.sort(key=lambda r: _reg_key_for_role_balanced(r, "C2", is_bad))
            c2 = c2_pool[0]
            by_day[d]["C2"] = c2.name
            c2.stats_by_role["C2"]["total"] += 1
            if is_bad:
                c2.stats_by_role["C2"]["bad"] += 1
            picked.add(c2.name)
    return by_day


def plan_ops_assignments(
    all_ops: List[date],
    bad_ops: Set[date],
    people: List[Person],
    team_size: int,
    target_bad: Dict[str, float],
    nstt_cfg: NSTTConfig,
) -> Tuple[Dict[date, List[str]], Dict[str, Dict[str, int]], Dict[date, Dict[str, str]]]:
    """
    Plan NSF assignments to Ops days and, if NSTT is enabled, assign NSTT
    stages. Translated from the original script. The function returns:
      - schedule: {date: [names]}
      - stats: {name: {"total": int, "bad": int}}
      - nstt_day_roles: {date: {name: "O", "E", "A", or combinations}}
    """
    # When NSTT is enabled the team size becomes 5 regardless of requested size
    if nstt_cfg.enabled:
        team_size = 5
    stats = {p.name: {"total": 0, "bad": 0} for p in people}
    schedule: Dict[date, List[str]] = {}
    nstt_day_roles: Dict[date, Dict[str, str]] = {}

    def eligible_for_saturday_oil(p: Person, d: date) -> bool:
        # Limit a person to one Ops in the same week if one is Saturday oil
        for od, team in schedule.items():
            if p.name in team and od.isocalendar()[1] == d.isocalendar()[1]:
                # Already on an Ops team this week
                return False
        return True

    for d in sorted(all_ops):
        is_bad = d in bad_ops
        # An NSF cannot serve Ops on d if they are constrained on d or d+1
        available = [
            p
            for p in people
            if (d not in p.unavailable_ops)
            and ((d + timedelta(days=1)) not in p.unavailable)
        ]
        if not available:
            schedule[d] = []
            continue

        def key(p: Person):
            bad_diff = stats[p.name]["bad"] - target_bad.get(p.name, 0.0)
            total = stats[p.name]["total"]
            oil_penalty = 0
            if d.weekday() == 5:  # Saturday
                if not eligible_for_saturday_oil(p, d):
                    oil_penalty = 1000
            # Bias to start NSTT early if relevant
            nstt_bias = 0
            if nstt_cfg.enabled and p.name in nstt_cfg.members:
                m = nstt_cfg.members[p.name]
                start = m.start_date or nstt_cfg.global_start
                can_start = (start is None) or (d >= start)
                if can_start:
                    pending_c2 = bool(m.remaining.get("C2"))
                    pending_e1 = bool(m.remaining.get("E1")) and not pending_c2
                    nxt = None
                    if pending_c2 and m.remaining["C2"]:
                        nxt = m.remaining["C2"][0]
                    elif pending_e1 and m.remaining["E1"]:
                        nxt = m.remaining["E1"][0]
                    if nxt == "O":
                        nstt_bias = -1.5
                    elif nxt == "E":
                        nstt_bias = -1.0
                    elif nxt == "A":
                        nstt_bias = -0.2
            return (
                bad_diff if is_bad else 0,
                nstt_bias,
                total,
                oil_penalty,
                p.name.lower(),
            )

        ranked = sorted(available, key=key)
        team: List[Person] = []
        noncleared = 0
        # Fill team up to team_size
        for cand in ranked:
            if not cand.cleared and noncleared >= 1:
                continue
            if not cand.cleared:
                noncleared += 1
            team.append(cand)
            if len(team) == team_size:
                break
        # Ensure at least one cleared teammate if one non‑cleared exists
        if noncleared == 1 and not any(t.cleared for t in team):
            # find a cleared replacement
            extra = next((p for p in ranked if p.cleared and p not in team), None)
            if extra:
                # replace the non‑cleared person
                for i in range(len(team) - 1, -1, -1):
                    if not team[i].cleared:
                        team[i] = extra
                        break
        schedule[d] = [p.name for p in team]
        for p in team:
            stats[p.name]["total"] += 1
            if is_bad:
                stats[p.name]["bad"] += 1
        # NSTT assignment
        if nstt_cfg.enabled:
            nb = 0
            # Determine number of boardings for the day
            if d in nstt_cfg.no_boarding_ops:
                nb = 0
            elif d in nstt_cfg.three_boarding_ops:
                nb = 3
            else:
                nb = 1 if d.weekday() == 6 else 2
            if nb > 0:
                # Determine who can start/continue
                def _can_start(name: str) -> bool:
                    member = nstt_cfg.members.get(name)
                    if not member:
                        return False
                    per_start = member.start_date or nstt_cfg.global_start
                    return per_start is None or d >= per_start
                def _next_need(nm: str) -> Optional[Tuple[str, str]]:
                    m = nstt_cfg.members.get(nm)
                    if not m:
                        return None
                    for t in ["C2", "E1"]:
                        if t in m.types and m.remaining.get(t):
                            return (t, m.remaining[t][0])
                    return None
                team_names = schedule[d]
                cands = [nm for nm in team_names if _can_start(nm) and _next_need(nm) is not None]
                def _prio(nm: str):
                    tns = _next_need(nm)
                    stage_rank = {"O": 0, "E": 1, "A": 2}.get(tns[1], 3) if tns else 3
                    return (stage_rank, stats[nm]["total"], nm.lower())
                cands.sort(key=_prio)
                chosen = cands[:2]
                if chosen:
                    per_name_tags: Dict[str, List[str]] = {nm: [] for nm in chosen}
                    def _assign_one(nm: str, desired: str) -> Optional[str]:
                        m = nstt_cfg.members[nm]
                        t = "C2" if ("C2" in m.types and m.remaining.get("C2")) else ("E1" if "E1" in m.types else None)
                        if not t or not m.remaining.get(t):
                            return None
                        nxt = m.remaining[t][0]
                        if desired != nxt:
                            return None
                        # remove this stage
                        m.remaining[t].pop(0)
                        return desired
                    b = 1
                    while b <= nb:
                        used_nonobs = False
                        for nm in chosen:
                            need = _next_need(nm)
                            if not need:
                                continue
                            if need[1] in ("O", "E", "A"):
                                if need[1] != "O":
                                    if used_nonobs:
                                        continue
                                    st = _assign_one(nm, need[1])
                                    if st:
                                        per_name_tags[nm].append(st)
                                        used_nonobs = True
                                else:
                                    st = _assign_one(nm, "O")
                                    if st:
                                        per_name_tags[nm].append(st)
                        b += 1
                    if per_name_tags:
                        nstt_day_roles[d] = {nm: "&".join(tags) for nm, tags in per_name_tags.items() if tags}
    return schedule, stats, nstt_day_roles


def plan_standby_with_leaders(
    all_ops: List[date],
    nsf_schedule: Dict[date, List[str]],
    people: List[Person],
    leaders_by_day: Dict[date, Dict[str, Optional[str]]],
    regs: List[Regular],
    window_start: date,
    window_end: date,
) -> Dict[date, List[str]]:
    """
    Compute standby teams for each day (one day before Ops) including leaders
    and NSFs. Taken from the original script, simplified for web.
    """
    from copy import deepcopy
    regs_by_name = {r.name: r for r in regs}
    by_name = {p.name: p for p in people}
    standby: Dict[date, List[str]] = {}
    def _sub_for_role(role: str, sday: date, picked: Set[str]) -> Optional[str]:
        if role == "ATL":
            atl_only = [r for r in regs if r.roles == {"ATL"} and sday not in r.unavailable and r.name not in picked]
            if atl_only:
                atl_only.sort(key=lambda r: (r.stats_by_role["TL"]["total"], r.stats_by_role["ATL"]["total"], r.name.lower()))
                return atl_only[0].name
            c2_atl = [r for r in regs if r.roles == {"C2", "ATL"} and sday not in r.unavailable and r.name not in picked]
            if c2_atl:
                c2_atl.sort(key=lambda r: (r.stats_by_role["TL"]["total"], r.stats_by_role["ATL"]["total"], r.name.lower()))
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
        today_leads = leaders_by_day.get(d, {"TL": None, "ATL": None, "C2": None})
        for role in ["TL", "ATL", "C2"]:
            nm = today_leads.get(role)
            if nm:
                r = regs_by_name.get(nm)
                if r and sday not in r.unavailable:
                    picks.append(nm)
                    picked_set.add(nm)
                else:
                    sub = _sub_for_role(role, sday, picked_set)
                    picks.append(f"({sub})" if sub else "")
                    if sub:
                        picked_set.add(sub)
            else:
                sub = _sub_for_role(role, sday, picked_set)
                picks.append(sub if sub else "")
                if sub:
                    picked_set.add(sub)
        # NSFs (3 total, max 1 non‑cleared)
        want_nsf = 3
        nsf_picks: List[str] = []
        noncleared_cnt = 0
        blocked_due_to_constraint = 0
        next_day_team = nsf_schedule.get(d, [])
        for nm in next_day_team:
            if len(nsf_picks) >= want_nsf:
                break
            p = by_name.get(nm)
            if not p:
                continue
            if sday in p.unavailable:
                blocked_due_to_constraint += 1
                continue
            if (not p.cleared) and noncleared_cnt >= 1:
                continue
            nsf_picks.append(nm)
            if not p.cleared:
                noncleared_cnt += 1
        if len(nsf_picks) < want_nsf:
            team_set = set(nsf_picks)
            for p in people:
                if len(nsf_picks) >= want_nsf:
                    break
                if p.name in team_set:
                    continue
                if sday in p.unavailable:
                    continue
                if p.name in next_day_team and (not p.cleared) and noncleared_cnt >= 1:
                    continue
                if (not p.cleared) and noncleared_cnt >= 1:
                    continue
                if blocked_due_to_constraint > 0:
                    nsf_picks.append(f"({p.name})")
                    blocked_due_to_constraint -= 1
                else:
                    nsf_picks.append(p.name)
                if not p.cleared:
                    noncleared_cnt += 1
        standby[sday] = picks + nsf_picks
    return standby


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
) -> Dict[str, any]:
    """
    Build table data structures suitable for rendering via HTML. This helper
    transforms the schedule dictionaries into lists of rows with strings.
    Returns a dict with keys: ops_rows, standby_rows, nsf_summary_rows,
    leader_summary_rows.
    """
    # Ops table
    ops_rows: List[List[str]] = []
    for d in sorted(all_ops):
        day_str = f"{d.day:02d} {d.strftime('%b').lower()} ({d.strftime('%a').lower()})"
        if d in bad_ops:
            # mark bad days with * for red; CSS will style
            day_str = f"*{day_str}*"
        leaders = leaders_by_day.get(d, {"TL": None, "ATL": None, "C2": None})
        nsf_names = nsf_schedule.get(d, [])
        nstt_tags = nstt_roles.get(d, {})
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
    # Standby table
    standby_rows: List[List[str]] = []
    for sday in sorted(standby.keys()):
        day_str = f"{sday.day:02d} {sday.strftime('%b').lower()} ({sday.strftime('%a').lower()})"
        team = standby[sday]
        tl = team[0] if len(team) > 0 else "—"
        atl = team[1] if len(team) > 1 else "—"
        c2 = team[2] if len(team) > 2 else "—"
        nsfs = team[3:] if len(team) > 3 else []
        nsf_str = ", ".join(nsfs) if nsfs else "—"
        standby_rows.append([day_str, tl or "—", atl or "—", c2 or "—", nsf_str])
        # NSF summary
    total_ops_count = len(all_ops)
    nsf_summary_rows: List[List[str]] = []
    for name in sorted(stats.keys(), key=str.lower):
        total = stats[name]["total"]
        bad = stats[name]["bad"]
        co = carry.get(name, 0)
        nsf_summary_rows.append([
            name,
            f"{total}/{total_ops_count}",
            str(bad),
            str(co),
        ])

    # Leader summary (compute from leaders_by_day instead of regs.stats_by_role)
    leader_summary_rows: List[List[str]] = []
    role_order = ["TL", "ATL", "C2"]

    # aggregate per leader per role using leaders_by_day + bad_ops
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

    # Emit rows grouped by role, with a blank spacer between groups
    for role in role_order:
        role_rows: List[List[str]] = []
        for name, roles in leader_stats.items():
            stats_for_role = roles.get(role)
            if not stats_for_role:
                continue
            role_rows.append([
                name,
                role,
                f"{stats_for_role['total']}/{total_ops_count}",
                str(stats_for_role["bad"]),
            ])
        role_rows.sort(key=lambda x: x[0].lower())
        if role_rows:
            if leader_summary_rows:  # spacer between role groups
                leader_summary_rows.append(["", "", "", ""])
            leader_summary_rows.extend(role_rows)

    return {
        "ops_rows": ops_rows,
        "standby_rows": standby_rows,
        "nsf_summary_rows": nsf_summary_rows,
        "leader_summary_rows": leader_summary_rows,
    }


# ====== Flask app setup ======

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "dev_secret_key_change_me")


def reset_session():
    """Clear planning data from session."""
    keys = [
        "start_date",
        "end_date",
        "anchor_date",
        "num_nsf",
        "num_leaders",
        "nsf_team_size",
        "three_boardings_any",
        "no_boardings_any",
        "nstt_enabled",
        "roster",
        "leaders",
        "nstt_cfg",
        "carryover",
    ]
    for k in keys:
        session.pop(k, None)


@app.route("/")
def index():
    """Entry point: ask for basic month details and counts."""
    return render_template("index.html")


@app.route("/step1", methods=["POST"])
def step1():
    """Handle submission from index and move to NSFs page."""
    try:
        start_date = parse_yyyy_mm_dd(request.form["start_date"])
        end_date = parse_yyyy_mm_dd(request.form["end_date"])
        anchor_date = parse_yyyy_mm_dd(request.form["anchor_date"])
    except Exception:
        flash("Please enter valid dates in YYYY‑MM‑DD format.")
        return redirect(url_for("index"))
    if end_date < start_date:
        flash("End date must not be before start date.")
        return redirect(url_for("index"))
    # Save to session
    session["start_date"] = start_date.isoformat()
    session["end_date"] = end_date.isoformat()
    session["anchor_date"] = anchor_date.isoformat()
    try:
        num_nsf = int(request.form["num_nsf"])
        num_leaders = int(request.form["num_leaders"])
    except Exception:
        flash("Please specify numeric values for number of NSFs and leaders.")
        return redirect(url_for("index"))
    if num_nsf <= 0 or num_leaders < 0:
        flash("Number of NSFs must be positive and leaders non‑negative.")
        return redirect(url_for("index"))
    session["num_nsf"] = num_nsf
    session["num_leaders"] = num_leaders
    session["nstt_enabled"] = "nstt_enabled" in request.form
    return redirect(url_for("nsfs"))


@app.route("/nsfs", methods=["GET", "POST"])
def nsfs():
    """Collect NSF roster and constraints."""
    if request.method == "POST":
        num = session.get("num_nsf")
        roster: List[Dict[str, any]] = []
        start_date = parse_yyyy_mm_dd(session["start_date"])
        end_date = parse_yyyy_mm_dd(session["end_date"])
        # iterate names
        for i in range(1, num + 1):
            name = request.form.get(f"nsf_name_{i}", "").strip()
            unavailable_str = request.form.get(f"nsf_unavail_{i}", "").strip()
            cleared = request.form.get(f"nsf_cleared_{i}") == "on"
            if not name:
                flash("All NSFs must have a name.")
                return redirect(url_for("nsfs"))
            unavailable_dates = _expand_dd_list_into_window_dates(unavailable_str, (start_date, end_date))
            roster.append({
                "name": name,
                "unavailable": [d.isoformat() for d in unavailable_dates],
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
    num_nsf = session.get("num_nsf")
    if not num_nsf:
        return redirect(url_for("index"))
    return render_template("nsfs.html", num_nsf=num_nsf)


@app.route("/leaders", methods=["GET", "POST"])
def leaders():
    """Collect leader (regular) information and Ops special cases."""
    if request.method == "POST":
        num_leaders = session.get("num_leaders")
        start_date = parse_yyyy_mm_dd(session["start_date"])
        end_date = parse_yyyy_mm_dd(session["end_date"])
        regs: List[Dict[str, any]] = []
        for i in range(1, num_leaders + 1):
            name = request.form.get(f"leader_name_{i}", "").strip()
            roles_raw = request.form.get(f"leader_roles_{i}", "").strip().upper().replace(" ", "")
            unavailable_str = request.form.get(f"leader_unavail_{i}", "").strip()
            if not name:
                flash("All leaders must have a name.")
                return redirect(url_for("leaders"))
            # Parse roles
            role_tokens = [tok for tok in roles_raw.split(",") if tok]
            roles = set(role_tokens)
            # Validate roles
            if roles not in ( {"TL"}, {"ATL"}, {"C2"}, {"C2", "ATL"} ):
                flash("Invalid roles for leader. Allowed: TL; ATL; C2; or C2,ATL.")
                return redirect(url_for("leaders"))
            unavailable_dates = _expand_dd_list_into_window_dates(unavailable_str, (start_date, end_date))
            regs.append({
                "name": name,
                "roles": list(roles),
                "unavailable": [d.isoformat() for d in unavailable_dates],
            })
        session["leaders"] = regs
        # special Ops markers
        three_any_raw = request.form.get("three_boardings_any", "").strip()
        no_any_raw = request.form.get("no_boardings_any", "").strip()
        three_dates = _expand_dd_list_into_window_dates(three_any_raw, (start_date, end_date))
        no_dates = _expand_dd_list_into_window_dates(no_any_raw, (start_date, end_date))
        session["three_boardings_any"] = [d.isoformat() for d in (three_dates or [])]
        session["no_boardings_any"] = [d.isoformat() for d in (no_dates or [])]
        # if NSTT enabled ask for NSTT; else ask for carryover and compute
        if session.get("nstt_enabled"):
            return redirect(url_for("nstt"))
        else:
            return redirect(url_for("carry"))
    num_leaders = session.get("num_leaders")
    if num_leaders is None:
        return redirect(url_for("index"))
    return render_template("leaders.html", num_leaders=num_leaders)


@app.route("/nstt", methods=["GET", "POST"])
def nstt():
    """Collect NSTT configuration if enabled."""
    if not session.get("nstt_enabled"):
        return redirect(url_for("carry"))
    if request.method == "POST":
        start_date = parse_yyyy_mm_dd(session["start_date"])
        end_date = parse_yyyy_mm_dd(session["end_date"])
        # global NSTT start (optional)
        gl_start_raw = request.form.get("nstt_global_start", "").strip()
        gl_start = parse_yyyy_mm_dd(gl_start_raw) if gl_start_raw else None
        # local 3/0 boardings for NSTT
        three_raw = request.form.get("nstt_three_boardings", "").strip()
        no_raw = request.form.get("nstt_no_boardings", "").strip()
        three_dates = _expand_dd_list_into_window_dates(three_raw, (start_date, end_date))
        no_dates = _expand_dd_list_into_window_dates(no_raw, (start_date, end_date))
        nstt_data: Dict[str, any] = {
            "enabled": True,
            "global_start": gl_start.isoformat() if gl_start else None,
            "three_boardings": [d.isoformat() for d in three_dates],
            "no_boardings": [d.isoformat() for d in no_dates],
            "members": [],
        }
        try:
            num_members = int(request.form.get("nstt_num_members", "0"))
        except Exception:
            num_members = 0
        # map roster names to canonical (case‑insensitive) names
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
            types = set(x for x in toks if x in {"C2", "E1"})
            # C2 stages
            rem = {}
            if "C2" in types:
                c2_rem_raw = request.form.get(f"nstt_c2_rem_{i}", "").strip().upper()
                if c2_rem_raw:
                    c2_toks = [t.strip() for t in c2_rem_raw.split(",") if t.strip()]
                    order = [x for x in ["O", "E", "A"] if x in c2_toks]
                    # If they specified O and E but left A, append A automatically
                    if "O" in order and "E" in order and "A" not in order:
                        order.append("A")
                    else:
                        order = order
                    rem["C2"] = order
            if "E1" in types:
                e1_rem_raw = request.form.get(f"nstt_e1_rem_{i}", "").strip().upper()
                if e1_rem_raw:
                    e1_toks = [t.strip() for t in e1_rem_raw.split(",") if t.strip()]
                    order = [x for x in ["O", "E", "A"] if x in e1_toks]
                    if "O" in order and "E" in order and "A" not in order:
                        order.append("A")
                    else:
                        order = order
                    rem["E1"] = order
            # personal start
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
    # GET
    return render_template("nstt.html")


@app.route("/carry", methods=["GET", "POST"])
def carry():
    """Collect carryover values and compute results."""
    if request.method == "POST":
        # parse carryover string, expecting Name:Number pairs
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
                            carry[p["name"]] = carry.get(p["name"], 0)
        session["carryover"] = carry
        return redirect(url_for("results"))
    return render_template("carry.html")


@app.route("/results")
def results():
    """Compute the schedule and render results tables."""
    # reconstruct objects from session
    required = ["start_date", "end_date", "anchor_date", "roster", "leaders"]
    missing = [k for k in required if k not in session]
    if missing:
        flash("Setup incomplete. Please start again.")
        return redirect(url_for("index"))
    start_date = parse_yyyy_mm_dd(session["start_date"])
    end_date = parse_yyyy_mm_dd(session["end_date"])
    anchor_date = parse_yyyy_mm_dd(session["anchor_date"])
    all_ops = generate_ops_dates_1_in_3(start_date, end_date, anchor_date)
    # NSFs
    roster_data = session.get("roster", [])
    people: List[Person] = []
    all_ops_set = set(all_ops)
    for p in roster_data:
        name = p["name"]
        unavailable = set(parse_yyyy_mm_dd(x) for x in p.get("unavailable", []))
        unavailable_ops = unavailable & all_ops_set
        cleared = p.get("cleared", False)
        people.append(Person(name=name, unavailable=unavailable, unavailable_ops=unavailable_ops, cleared=cleared))
    team_size = session.get("nsf_team_size", 3)
    # Leaders
    regs_data = session.get("leaders", [])
    regs: List[Regular] = []
    for r in regs_data:
        name = r["name"]
        roles = set(r.get("roles", []))
        unavailable = set(parse_yyyy_mm_dd(x) for x in r.get("unavailable", []))
        regs.append(Regular(name=name, roles=roles, unavailable=unavailable))
    # Special ops markers (global)
    three_boardings_any = set(parse_yyyy_mm_dd(x) for x in session.get("three_boardings_any", []))
    no_boardings_any = set(parse_yyyy_mm_dd(x) for x in session.get("no_boardings_any", []))
    # NSTT config
    nstt_cfg_data = session.get("nstt_cfg")
    if nstt_cfg_data:
        nstt_cfg = NSTTConfig(enabled=True)
        gs = nstt_cfg_data.get("global_start")
        nstt_cfg.global_start = parse_yyyy_mm_dd(gs) if gs else None
        nstt_cfg.three_boarding_ops = set(parse_yyyy_mm_dd(x) for x in nstt_cfg_data.get("three_boardings", []))
        nstt_cfg.no_boarding_ops = set(parse_yyyy_mm_dd(x) for x in nstt_cfg_data.get("no_boardings", []))
        for m in nstt_cfg_data.get("members", []):
            nm = m["name"]
            types = set(m.get("types", []))
            remaining: Dict[str, List[str]] = {}
            for k, v in m.get("remaining", {}).items():
                remaining[k] = list(v)
            pstart = m.get("start")
            st = parse_yyyy_mm_dd(pstart) if pstart else None
            nstt_cfg.members[nm] = NSTTMember(name=nm, types=types, remaining=remaining, start_date=st)
    else:
        nstt_cfg = NSTTConfig(enabled=False)
    # compute bad ops
    bad_ops = compute_auto_bad_ops(
        all_ops,
        three_boardings_any,
        no_boardings_any,
        nstt_cfg.enabled,
        nstt_cfg.three_boarding_ops,
        nstt_cfg.no_boarding_ops,
    )
    # carryover
    carry = session.get("carryover", {})
    # compute targets
    target_bad = compute_bad_targets(people, bad_ops, team_size, carry)
    # plan NSF assignments
    nsf_schedule, stats, nstt_roles = plan_ops_assignments(
        all_ops, bad_ops, people, team_size, target_bad, nstt_cfg
    )
    # plan leaders
    leaders_by_day = plan_leaders_for_ops(all_ops, regs, bad_ops)
    # standby
    standby = plan_standby_with_leaders(all_ops, nsf_schedule, people, leaders_by_day, regs, start_date, end_date)
    # build tables
    tables = build_tables(all_ops, bad_ops, leaders_by_day, nsf_schedule, nstt_roles, stats, carry, standby, regs)
    # Render results
    return render_template(
        "results.html",
        ops_rows=tables["ops_rows"],
        standby_rows=tables["standby_rows"],
        nsf_summary_rows=tables["nsf_summary_rows"],
        leader_summary_rows=tables["leader_summary_rows"],
    )


@app.route("/reset")
def reset():
    """Clear session and return to start."""
    reset_session()
    return redirect(url_for("index"))


if __name__ == "__main__":
    # When running directly, start the development server on port 8000 for
    # compatibility with environments that expose that port publicly. The
    # threaded option allows multiple concurrent users during development.
    app.run(host="0.0.0.0", port=8000, debug=True)
