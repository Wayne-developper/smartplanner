# SmartPlanner (MVP)
An intelligent day planner that schedules tasks **inside hard work windows** using a **backtracking (DFS)** engine.  
This repository contains the **public core**: the planner algorithm and a small Python API.  
Product features (full app UI, persistence schema, business roadmap) are intentionally **kept private**.

> **Status:** MVP – usable, tested with smoke tests, actively evolving.
>
---

## Why SmartPlanner?

- **Hard guarantees**: nothing goes outside the time you set (e.g., 15:00–20:00).
- **Flexible preferences**: soft deadlines and optional breaks influence the score, never the rules.
- **Practical control**: priorities, “must do”, seed-based variants, and **lock & replan**.

---

## What’s in this repo (public)

- `TimeManager` class (Python) with:
  - Hard **work windows** + no overlaps
  - **Soft / hard deadlines**
  - **Break policy** (optional breaks, preferred interval, caps)
  - **Subset selection** when capacity is tight
  - **Seed** for deterministic alternative plans
  - **Lock & replan** to fix specific tasks at specific times
  - Simple branch-and-bound pruning for performance

### What’s intentionally private (not in this repo)

- Full application UI & UX flows
- Backend persistence layer design beyond MVP (complete SQL schema)
- Internal heuristics/product strategy

---

## Quick start

Requirements: Python 3.10+ (tested on 3.12). No external deps for the core.

```bash
# clone
git clone https://github.com/WAYNE-developper/smartplanner.git
cd smartplanner

# (optional) create venv
# python -m venv .venv && . .venv/Scripts/activate   # Windows
# python3 -m venv .venv && source .venv/bin/activate # macOS/Linux

# **Minimal example**
from datetime import datetime
from console_global import TimeManager  # adjust import if your file name differs

tm = TimeManager(time_unit=15)
tm.add_work_window(tm.hm_to_min("12:00"), tm.hm_to_min("17:00"))

tm.set_break_policy(allowed_durations=[15, 30], preferred_interval_minutes=90, max_total_break_minutes=45, max_breaks=2)

# No deadlines needed; priority drives the score
t1 = tm.add_task("Study",    duration=60,  priority=5)
t2 = tm.add_task("Exercises",duration=90,  priority=4)
t3 = tm.add_task("Project",  duration=120, priority=3)
t4 = tm.add_task("Email",    duration=45,  priority=2)

# Plan (seed gives an alternative valid variant)
result = tm.plan(seed=1337)

print(result["stats"], "score:", result["score_total"])
for item in result["plan_items"]:
    print(item)
print("unplanned:", result["unplanned"])

# ** Lock & replan example **
locked = [{"task_id": t1, "start_min": tm.hm_to_min("12:00")}]
locked_plan = tm.lock_and_replan(locks=locked, seed=7)

# ** Smoke test **
from datetime import datetime
from console_global import TimeManager

def assert_invariants(result, work_windows, time_unit):
    items = result["plan_items"]
    for it in items:
        s, e = it["start_min"], it["end_min"]
        assert any(s >= w0 and e <= w1 for (w0, w1) in work_windows), f"Outside windows: {it}"
        assert s % time_unit == 0 and e % time_unit == 0, f"Misaligned: {it}"
    occ = sorted((it["start_min"], it["end_min"], it["kind"]) for it in items)
    for i in range(1, len(occ)):
        assert occ[i-1][1] <= occ[i][0], f"Overlap: {occ[i-1]} vs {occ[i]}"

def smoke_basic():
    tm = TimeManager(time_unit=15)
    w = [(tm.hm_to_min("12:00"), tm.hm_to_min("17:00"))]
    for s,e in w: tm.add_work_window(s, e)
    tm.set_break_policy()
    t1 = tm.add_task("A", 60, 5)
    tm.add_task("B", 90, 4)
    tm.add_task("C", 120, 3)
    r = tm.plan()
    assert_invariants(r, w, tm.time_unit)

def smoke_hard_deadline():
    tm = TimeManager(time_unit=15)
    w = [(tm.hm_to_min("09:00"), tm.hm_to_min("12:00"))]
    for s,e in w: tm.add_work_window(s, e)
    dl = datetime(2025, 1, 1, 11, 0, 0)
    t = tm.add_task("Hard", 45, 5, deadline=dl, deadline_hard=True)
    r = tm.plan()
    assert_invariants(r, w, tm.time_unit)
    end = [it["end_min"] for it in r["plan_items"] if it["task_id"] == t][0]
    assert end <= tm.hm_to_min("11:00")

if __name__ == "__main__":
    smoke_basic()
    smoke_hard_deadline()
    print("All smoke tests passed ✅")

