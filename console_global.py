from typing import Optional, List, Tuple, Dict
from datetime import datetime
import itertools
import time
import random
from itertools import groupby

class TimeManager:
    def __init__(self, time_unit: int = 15):
        self.tasks: List[Dict] = []
        self.work_windows: List[Tuple[int, int]] = []
        self.break_policy: Optional[Dict] = None
        self.time_unit = time_unit
        self._id_counter = itertools.count(1)  # For unique IDs

        # Default weights (tunable via set_weights)
        self.weights = {
            "w_prio": 10.0,
            "w_early": 0.05,      # bonus per minute finished before deadline (capped)
            "w_late": 0.20,       # penalty per minute after a soft deadline
            "w_break_bonus": 1.0  # small bonus when a recommended break fits
        }

    # --- Helpers ---
    @staticmethod
    def hm_to_min(hhmm: str) -> int:
        """'09:30' -> 570 (minutes since midnight)."""
        h, m = map(int, hhmm.split(":"))
        return h * 60 + m

    @staticmethod
    def dt_to_min(dt: datetime) -> int:
        """datetime -> minutes since midnight (single-day MVP; date is ignored)."""
        return dt.hour * 60 + dt.minute

    def _next_id(self) -> int:
        return next(self._id_counter)

    # --- Data API ---
    def add_task(
        self,
        title: str,
        duration: int,
        priority: int,
        deadline: Optional[datetime] = None,
        *,
        deadline_hard: bool = False,
        must_do: bool = False
    ) -> int:
        # Validation
        if not isinstance(title, str) or not title.strip():
            raise ValueError("title must be a non-empty string")
        if not isinstance(duration, int) or duration <= 0:
            raise ValueError("duration must be a positive integer (minutes)")
        if duration % self.time_unit != 0:
            raise ValueError(f"duration must be a multiple of time_unit={self.time_unit} minutes")
        if not isinstance(priority, int) or not (1 <= priority <= 5):
            raise ValueError("priority must be an integer in [1, 5]")
        if deadline is not None and not isinstance(deadline, datetime):
            raise TypeError("deadline must be a datetime or None")
        if not isinstance(deadline_hard, bool) or not isinstance(must_do, bool):
            raise TypeError("deadline_hard and must_do must be boolean")

        task = {
            "id": self._next_id(),
            "title": title.strip(),
            "duration": duration,
            "priority": priority,
            "deadline": deadline,
            "deadline_hard": deadline_hard,
            "must_do": must_do
        }
        self.tasks.append(task)
        return task["id"]

    def add_work_window(self, start_time: int, end_time: int):
        """start_time / end_time in minutes (0..1440), end is exclusive logically."""
        if not (0 <= start_time < end_time <= 24 * 60):
            raise ValueError("start_time/end_time must be within [0..1440] and start < end")
        if (start_time % self.time_unit) or (end_time % self.time_unit):
            raise ValueError(f"times must be aligned to time_unit={self.time_unit} minutes")
        # Prevent overlapping windows
        for s, e in self.work_windows:
            if not (end_time <= s or start_time >= e):
                raise ValueError("this work window overlaps an existing window")
        self.work_windows.append((start_time, end_time))

    def set_break_policy(
        self,
        break_policy: Optional[Dict] = None,
        *,
        allowed_durations: Optional[List[int]] = None,
        preferred_interval_minutes: Optional[int] = None,
        max_total_break_minutes: Optional[int] = None,
        max_breaks: Optional[int] = None,
        weight_bonus: Optional[float] = None
    ):
        """
        Two ways to configure:
        - pass a full dict in break_policy
        - or pass individual keyword arguments (recommended initially)
        """
        if break_policy is None:
            bp = {
                "allowed_durations": allowed_durations or [15, 30],
                "preferred_interval_minutes": preferred_interval_minutes or 90,
                "max_total_break_minutes": max_total_break_minutes or 60,
                "max_breaks": max_breaks or 3,
                "weight_bonus": float(weight_bonus) if weight_bonus is not None else 1.0,
            }
        else:
            bp = break_policy

        # Policy validation
        if any(d <= 0 for d in bp["allowed_durations"]):
            raise ValueError("allowed_durations must contain durations > 0")
        if any(d % self.time_unit != 0 for d in bp["allowed_durations"]):
            raise ValueError(f"allowed_durations must be multiples of time_unit={self.time_unit}")
        if bp["preferred_interval_minutes"] <= 0:
            raise ValueError("preferred_interval_minutes must be > 0")
        if bp["max_total_break_minutes"] < 0 or bp["max_breaks"] < 0:
            raise ValueError("max_total_break_minutes and max_breaks must be >= 0")

        self.break_policy = bp

    # --- Settings ---
    def set_time_unit(self, time_unit: int):
        """Allowed only while there are no tasks, no work windows, and no break policy defined."""
        if not isinstance(time_unit, int) or time_unit <= 0:
            raise ValueError("time_unit must be a positive integer")
        if 60 % time_unit != 0:
            raise ValueError("time_unit must divide 60 (e.g., 5, 10, 15, 20, 30, 60)")
        if self.tasks or self.work_windows or self.break_policy is not None:
            raise RuntimeError("changing time_unit is only allowed before adding tasks, work windows, or a break policy")
        self.time_unit = time_unit

    def set_weights(self, **kwargs):
        """Example: set_weights(w_prio=12, w_late=0.3)"""
        for k, v in kwargs.items():
            if k not in self.weights:
                raise KeyError(f"unknown weight: {k}")
            if not isinstance(v, (int, float)):
                raise ValueError("weights must be numeric")
            self.weights[k] = float(v)

    # --- Planning (DFS / Backtracking) ---
    def plan(
        self,
        time_limit_ms: Optional[int] = 1500,
        node_cap: Optional[int] = 200000,
        *,
        seed: Optional[int] = None,
        locks: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Build the best plan under hard constraints:
          - work windows (tasks and breaks must fully fit inside),
          - no overlaps,
          - hard deadlines respected.
        Soft constraints (soft deadlines, break preferences) contribute to the score but never violate hard rules.

        Returns
        -------
        dict with:
          - plan_items: [{task_id, start_min, end_min, kind: "task"|"break"}]
          - score_total: float
          - stats: {nodes, prunes, runtime_ms}
          - unplanned: [{task_id, reason}]
        """
        if not self.work_windows:
            return {
                "plan_items": [],
                "score_total": 0.0,
                "stats": {"nodes": 0, "prunes": 0, "runtime_ms": 0},
                "unplanned": [{"task_id": t["id"], "reason": "no_work_windows"} for t in self.tasks]
            }

        start_t = time.time()
        rng = random.Random(seed)

        # --- Prepare slots and window masks ---
        unit = self.time_unit
        num_slots = (24 * 60) // unit

        is_work_slot = [False] * num_slots
        windows_slots: List[Tuple[int, int]] = []
        for s_min, e_min in sorted(self.work_windows):
            s_slot = s_min // unit
            e_slot = e_min // unit
            for k in range(s_slot, e_slot):
                is_work_slot[k] = True
            windows_slots.append((s_slot, e_slot))  # end exclusive

        # Timeline: -1 free, -2 break, >=1 task_id
        timeline = [-1] * num_slots

        # --- Pre-checks: capacity and max window length ---
        max_window_len_slots = max((e - s) for s, e in windows_slots)

        unplanned: List[Dict] = []
        tasks_to_plan: List[Dict] = []

        # Map for locks
        by_id = {t["id"]: t for t in self.tasks}

        for t in self.tasks:
            t_slots = t["duration"] // unit
            if t_slots > max_window_len_slots:
                unplanned.append({"task_id": t["id"], "reason": "too_long_for_any_window"})
            else:
                tasks_to_plan.append({
                    **t,
                    "slots": t_slots,
                    "deadline_slot": None if t["deadline"] is None else (self.dt_to_min(t["deadline"]) // unit)
                })

        # --- Place locked tasks first (hard) ---
        locked_items: List[Tuple[int, int, int, str]] = []  # (task_id, start_slot, end_slot, "task")
        locked_ids = set()
        if locks:
            for lk in locks:
                if "task_id" not in lk or "start_min" not in lk:
                    raise ValueError("each lock must include 'task_id' and 'start_min'")
                tid = lk["task_id"]
                if tid not in by_id:
                    raise ValueError(f"locked task_id {tid} does not exist")
                if tid in locked_ids:
                    raise ValueError(f"duplicate lock for task_id {tid}")
                t = by_id[tid]
                t_slots = t["duration"] // unit
                if lk["start_min"] % unit != 0:
                    raise ValueError(f"locked start_min must align to time_unit={unit} minutes")
                start_slot = lk["start_min"] // unit
                end_slot = start_slot + t_slots

                # Must fit entirely within a single work window
                if not any(ws <= start_slot and end_slot <= we for ws, we in windows_slots):
                    raise ValueError(f"locked task {tid} does not fit entirely within a single work window")

                # Must be on work and free slots
                for s in range(start_slot, end_slot):
                    if not is_work_slot[s] or timeline[s] != -1:
                        raise ValueError(f"locked task {tid} overlaps non-work or occupied slot")

                # Place locked task
                for s in range(start_slot, end_slot):
                    timeline[s] = tid
                locked_items.append((tid, start_slot, end_slot, "task"))
                locked_ids.add(tid)

            # Remove locked tasks from the planning set
            tasks_to_plan = [tt for tt in tasks_to_plan if tt["id"] not in locked_ids]

        # --- Order tasks (deadline soonest, then higher priority, then shorter) ---
        def task_key(tt):
            dl = 10**9 if tt["deadline_slot"] is None else tt["deadline_slot"]
            return (dl, -tt["priority"], tt["slots"])

        tasks_to_plan.sort(key=task_key)

        # Deterministic diversification inside equal-key groups (when seed is set)
        if seed is not None and tasks_to_plan:
            grouped = []
            for _, grp in groupby(tasks_to_plan, key=task_key):
                g = list(grp)
                rng.shuffle(g)
                grouped.extend(g)
            tasks_to_plan = grouped

        # --- Stats & optimistic bound ---
        nodes = 0
        prunes = 0
        best_score = float("-inf")
        best_items: List[Tuple[int, int, int, str]] = locked_items[:]  # keep locked items
        best_ids_set = {tid for (tid, _, _, kind) in locked_items if kind == "task"}

        def optimistic_gain(tt):
            return self.weights["w_prio"] * tt["priority"]

        optimistic_suffix = [0.0] * (len(tasks_to_plan) + 1)
        for i in range(len(tasks_to_plan) - 1, -1, -1):
            optimistic_suffix[i] = optimistic_suffix[i + 1] + optimistic_gain(tasks_to_plan[i])

        # --- Placement utilities ---
        def block_free(start, end) -> bool:
            for s in range(start, end):
                if not is_work_slot[s] or timeline[s] != -1:
                    return False
            return True

        def place_block(start, end, mark):
            for s in range(start, end):
                timeline[s] = mark

        def feasible_starts_for(tt):
            L = tt["slots"]
            for ws, we in windows_slots:
                if we - ws < L:
                    continue
                s = ws
                while s + L <= we:
                    if block_free(s, s + L):
                        if tt["deadline_hard"] and tt["deadline_slot"] is not None:
                            if s + L <= tt["deadline_slot"]:
                                yield s
                        else:
                            yield s
                    s += 1

        def pause_choices(slots_since_break, breaks_used, break_min_total, end_slot):
            if self.break_policy is None:
                return []
            bp = self.break_policy
            pref_slots = bp["preferred_interval_minutes"] // unit
            if slots_since_break < pref_slots:
                return []
            if breaks_used >= bp["max_breaks"]:
                return []
            choices = []
            for dmin in bp["allowed_durations"]:
                dslots = dmin // unit
                # Find the window containing end_slot
                for ws, we in windows_slots:
                    if ws <= end_slot <= we:
                        if end_slot + dslots <= we and block_free(end_slot, end_slot + dslots):
                            if break_min_total + dmin <= bp["max_total_break_minutes"]:
                                choices.append((dslots, dmin))
                        break
            return choices

        def gain_for_task(tt, end_slot):
            g = self.weights["w_prio"] * tt["priority"]
            if tt["deadline_slot"] is not None:
                margin_slots = tt["deadline_slot"] - end_slot
                if margin_slots >= 0:
                    margin_min = margin_slots * unit
                    g += self.weights["w_early"] * min(margin_min, 240)  # cap at 4h early
                else:
                    late_min = (-margin_slots) * unit
                    g -= self.weights["w_late"] * late_min
            return g

        must_do_count = sum(1 for t in tasks_to_plan if t["must_do"])

        # --- DFS ---
        def dfs(i, score, plan_items, slots_since_break, breaks_used, break_min_total, must_do_remaining):
            nonlocal nodes, prunes, best_score, best_items, best_ids_set
            nodes += 1
            if node_cap is not None and nodes > node_cap:
                return
            if time_limit_ms is not None and (time.time() - start_t) * 1000.0 > time_limit_ms:
                return

            # All tasks considered
            if i == len(tasks_to_plan):
                if must_do_remaining:
                    return
                if score > best_score:
                    best_score = score
                    best_items = plan_items[:]
                    best_ids_set = {tid for (tid, _, _, kind) in best_items if kind == "task"}
                return

            # Optimistic bound
            if score + optimistic_suffix[i] < best_score:
                prunes += 1
                return

            tt = tasks_to_plan[i]
            placed_somewhere = False

            # Try all feasible placements (ASAP order)
            for start in feasible_starts_for(tt):
                end = start + tt["slots"]
                place_block(start, end, tt["id"])
                placed_somewhere = True
                new_score = score + gain_for_task(tt, end)

                # Branch with a break (optional / soft)
                for dslots, dmin in pause_choices(slots_since_break + tt["slots"], breaks_used, break_min_total, end):
                    place_block(end, end + dslots, -2)
                    dfs(
                        i + 1,
                        new_score + self.break_policy["weight_bonus"],  # small bonus
                        plan_items + [(tt["id"], start, end, "task"), (-1, end, end + dslots, "break")],
                        0,  # reset since break
                        breaks_used + 1,
                        break_min_total + dmin,
                        must_do_remaining - (1 if tt["must_do"] else 0)
                    )
                    place_block(end, end + dslots, -1)

                # Branch without break
                dfs(
                    i + 1,
                    new_score,
                    plan_items + [(tt["id"], start, end, "task")],
                    slots_since_break + tt["slots"],
                    breaks_used,
                    break_min_total,
                    must_do_remaining - (1 if tt["must_do"] else 0)
                )

                # backtrack the task
                place_block(start, end, -1)

            # Skip-task branch (choose a subset) — forbidden if must_do
            if not tt["must_do"]:
                dfs(
                    i + 1,
                    score,
                    plan_items,
                    slots_since_break,
                    breaks_used,
                    break_min_total,
                    must_do_remaining
                )
            else:
                if not placed_somewhere:
                    prunes += 1
                    return

        dfs(
            i=0,
            score=0.0,
            plan_items=locked_items[:],  # keep locked items in base
            slots_since_break=0,
            breaks_used=0,
            break_min_total=0,
            must_do_remaining=must_do_count
        )

        runtime_ms = int((time.time() - start_t) * 1000.0)
        plan_items_out = []
        planned_ids = set()
        for tid, s, e, kind in best_items:
            if kind == "task":
                planned_ids.add(tid)
            plan_items_out.append({
                "task_id": None if kind == "break" else tid,
                "start_min": s * unit,
                "end_min": e * unit,
                "kind": kind
            })

        # Unplanned tasks: too long for any window (pre-check) + not selected
        for t in tasks_to_plan:
            if t["id"] not in planned_ids:
                unplanned.append({"task_id": t["id"], "reason": "not_selected_or_no_slot"})

        return {
            "plan_items": plan_items_out,
            "score_total": 0.0 if best_score == float("-inf") else best_score,
            "stats": {"nodes": nodes, "prunes": prunes, "runtime_ms": runtime_ms},
            "unplanned": unplanned
        }

    # Convenience wrapper
    def lock_and_replan(self, locks: List[Dict], **kwargs) -> Dict:
        """Re-plan with some tasks locked in place. Example lock: {'task_id': 3, 'start_min': 540}."""
        return self.plan(locks=locks, **kwargs)
