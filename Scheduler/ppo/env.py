import numpy as np
import gymnasium as gym
import logging
from gymnasium.spaces import MultiDiscrete, Box
from collections import defaultdict
from models.Course import Course
from models.Faculty import Faculty
from models.Classroom import Classroom
from models.Timetable import Timetable, TimetableEntry, ClassTimetable
import random
import os
from gymnasium.utils import seeding


# Configure Logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

env_logger = logging.getLogger("TimetableEnv")
env_logger.setLevel(logging.INFO)

# Only add handlers once
if not env_logger.handlers:
    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, "env.log"))
    fh.setLevel(logging.INFO)

    # Formatter
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)

    env_logger.addHandler(fh)

    # Prevent messages from propagating to the root logger
    env_logger.propagate = False


class TimetableEnv(gym.Env):
    def __init__(self, timetable: Timetable, max_steps=100):
        super().__init__()
        env_logger.info("Initializing TimetableEnv...")
        self.timetable = timetable

        # dimensions
        self.time_slots        = [ts for ts in timetable.time_slots if ts != "12:00-13:00"]
        self.num_courses       = len(timetable.courses)
        self.num_days          = len(timetable.days)
        self.num_slots_per_day = len(self.time_slots)
        self.num_classrooms    = len(timetable.classrooms)
        self.num_flat_slots    = self.num_days * self.num_slots_per_day
        self.max_steps         = max_steps

        env_logger.info(f"Timetable dimensions: {self.num_courses} courses, "
                        f"{self.num_days} days, {self.num_slots_per_day} slots/day, "
                        f"{self.num_classrooms} classrooms.")

        # action & observation spaces
        self.action_space = MultiDiscrete((
            self.num_courses+1,
            self.num_flat_slots,
            self.num_classrooms,
        ))
        self.observation_space = Box(
            low=0,
            high=self.num_courses,
            shape=(self.num_flat_slots * self.num_classrooms,),
            dtype=np.int32,
        )

        # conflict trackers
        self.faculty_schedule   = defaultdict(set)  # (day, slot_str)
        self.classroom_schedule = defaultdict(set)  # (day, slot_str)
        self.course_day_scheduled = {i:set() for i in range(self.num_courses)}
        # per-course session counts
        # each value is {'theory': int, 'lab': int}
        self.sessions_scheduled = defaultdict(lambda: {'theory': 0, 'lab': 0})

        self.reset()

    def is_valid_action(self, course_index, day, time_slot, classroom):

        """Check all constraints before scheduling."""
        env_logger.debug(f"Validating action: course_index={course_index}, day={day}, "
                         f"time_slot={time_slot}, classroom={classroom}")

        course = self.timetable.courses[course_index]
        credits     = course.credits
        ctype       = course.subject_type  # "theory" or "lab"
        faculty_id  = course.faculty_id
        classroom_obj = self.timetable.classrooms[classroom]
        room_type   = classroom_obj.type     # "theory" or "lab"
        slot_idx = self.time_slots.index(time_slot)

        # —— 1) Session‑count constraints ——
        if ctype == "theory":
            # theory courses: need exactly `credits` sessions
            if self.sessions_scheduled[course_index]['theory'] >= credits:
                env_logger.warning(f"Invalid action: Theory sessions for course {course_index} "
                                   f"already scheduled.")
                return False, -50

        else:  # lab course
            # lab session?
            is_lab_session = (room_type == "lab")
            if is_lab_session:
                # only 1 lab session
                if self.sessions_scheduled[course_index]['lab'] >= 1:
                    env_logger.warning(f"Invalid action: Lab sessions for course {course_index} "
                                       f"already scheduled.")
                    return False, -50
                # must fit two consecutive slots
                if slot_idx >= self.num_slots_per_day - 1:
                    env_logger.warning("Invalid action: Lab session cannot fit into the last slot.")
                    return False, -50
            else:
                # theory part of a lab course: credits-1 sessions
                if self.sessions_scheduled[course_index]['theory'] >= (credits - 1):
                    env_logger.warning(f"Invalid action: Theory sessions for course {course_index} "
                                       f"already scheduled.")
                    return False, -50

        # —— 2) Faculty & classroom conflict checks ——
        # single‑slot check
        if (day, time_slot) in self.faculty_schedule[faculty_id]:
            env_logger.warning(f"Invalid action: Faculty conflict for {faculty_id} at {day}, {time_slot}.")
            return False, -50
        if (day, time_slot) in self.classroom_schedule[classroom_obj.code]:
            env_logger.warning(f"Invalid action: Classroom conflict for {classroom_obj.code} at {day}, {time_slot}.")
            return False, -50

        # consecutive faculty‑break check
        if slot_idx > 0:
            prev_slot = self.time_slots[slot_idx - 1]
            if (day, prev_slot) in self.faculty_schedule[faculty_id]:
                env_logger.warning(f"Faculty has a break just before/after the selected slot.")
                return False, -50

        if slot_idx < self.num_slots_per_day - 1:
            next_slot = self.time_slots[slot_idx + 1]
            if (day, next_slot) in self.faculty_schedule[faculty_id]:
                env_logger.warning(f"Faculty has a break just before/after the selected slot.")
                return False, -50

        # for lab session, also check the *second* slot
        if ctype == "lab" and room_type == "lab":
            # slot_idx < num_slots_per_day - 1 is already guaranteed above
            second_slot = self.time_slots[slot_idx + 1]
            if (day, second_slot) in self.faculty_schedule[faculty_id] or \
            (day, second_slot) in self.classroom_schedule[classroom_obj.code]:
                env_logger.warning(f"Faculty has a break just before/after the selected slot.")
                return False, -50

        # lunch break
        if time_slot == "12:00-13:00":
            env_logger.warning(f"Invalid action: Lunch break at {day}, {time_slot}.")
            return False, -50

        # NEW: Course-per-day check
        if day in self.course_day_scheduled[course_index]:
            env_logger.warning(f"Invalid action: Course {course_index} already scheduled on {day}.")
            return False, -50

        # edge‑slot penalty
        if time_slot in {"09:00-50:00", "15:00-16:00"}:
            env_logger.warning(f"Invalid action: Edge slot at {day}, {time_slot}.")
            return True, -2
        env_logger.info(f"Action is valid: course_index={course_index}, day={day}, "
                        f"time_slot={time_slot}, classroom={classroom}")
        return True, 1
    def step(self, action):
        """Take a step in the environment.
        action = (course_index, flat_ts, classroom)
        """
        env_logger.info(f"Taking step with action: {action}")

        # 1) Unpack and cast to native ints
        course_index, flat_ts, classroom = action
        course_index = int(course_index)
        flat_ts      = int(flat_ts)
        classroom    = int(classroom)

        # 2) Reserve 0 = no course; heavy penalty
        if course_index == 0:
            return self.state.flatten(), -50, False, False, {}

        # 3) Map to real course idx 0..num_courses-1
        ci = course_index - 1

        # Validate indices
        if not (0 <= ci < self.num_courses):
            env_logger.error(f"Invalid course index after shift: {ci}")
            return self.state.flatten(), -50, False, False, {}

        self.current_step += 1
        reward = 0
        done   = False

        # Decode day & slot
        day_idx   = flat_ts // self.num_slots_per_day
        slot_idx  = flat_ts % self.num_slots_per_day
        day       = self.timetable.days[day_idx]
        time_slot = self.time_slots[slot_idx]

        # 4) If slot occupied, find next
        if self.state[flat_ts, classroom] != 0:
            env_logger.info(f"Slot busy at {flat_ts},{classroom}, finding next valid.")
            next_flat, next_day, next_slot, term = self.find_next_valid(ci, flat_ts, classroom)
            if next_flat is None:
                return self.state.flatten(), term, False, False, {}
            flat_ts, day, time_slot = next_flat, next_day, next_slot
            reward += term

        # 5) Course-per-day check (use ci)
        if day in self.course_day_scheduled[ci]:
            return self.state.flatten(), -50, False, False, {}

        # 6) Validate constraints
        valid, penalty = self.is_valid_action(ci, day, time_slot, classroom)
        if not valid:
            return self.state.flatten(), penalty, False, False, {}
        reward += penalty

        # 7) Perform scheduling
        course        = self.timetable.courses[ci]
        faculty       = next(f for f in self.timetable.faculty if f.short_name == course.faculty_id)
        classroom_obj = self.timetable.classrooms[classroom]

        # Branch timetable lookup
        branch_sem = next(
            f"{b.branch_name}&{b.semester}"
            for b in self.timetable.branches
            if course in b.courses
        )
        class_tt = self.timetable.timetables[branch_sem]

        # Entry and RL state
        entry = TimetableEntry(day, time_slot, course, faculty, classroom_obj)
        self.state[flat_ts, classroom] = course_index  # store shifted index

        # Update trackers
        self.course_day_scheduled[ci].add(day)
        if course.subject_type == "lab" and classroom_obj.type == "lab":
            # two-slot lab
            next_flat_ts = flat_ts + 1
            next_slot    = self.time_slots[slot_idx + 1]
            self.state[next_flat_ts, classroom] = course_index
            self.course_day_scheduled[ci].add(day)
            self.sessions_scheduled[ci]['lab'] += 1
            # fill both slots in timetables...
            class_tt.timetable[day][time_slot] = entry
            class_tt.timetable[day][next_slot]  = entry
            self.faculty_schedule[faculty.short_name].update({(day, time_slot), (day, next_slot)})
            self.classroom_schedule[classroom_obj.code].update({(day, time_slot), (day, next_slot)})
        else:
            # single-slot theory
            self.sessions_scheduled[ci]['theory'] += 1
            class_tt.timetable[day][time_slot] = entry
            self.faculty_schedule[faculty.short_name].add((day, time_slot))
            self.classroom_schedule[classroom_obj.code].add((day, time_slot))

        # 8) Check done
        if np.count_nonzero(self.state) >= self.num_courses or self.current_step >= self.max_steps:
            done = True
            env_logger.info("Episode done.")

        return self.state.flatten(), reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        env_logger.info("Resetting environment...")

        for i in range(self.num_courses):
            self.course_day_scheduled[i].clear()

        # 1) RL state
        self.state = np.zeros((self.num_flat_slots, self.num_classrooms), dtype=np.int32)
        self.current_step = 0

        # 2) conflict trackers
        self.faculty_schedule.clear()
        self.classroom_schedule.clear()

        # 3) session counts
        self.sessions_scheduled.clear()

        # 4) clear underlying Timetable
        for class_tt in self.timetable.timetables.values():
            for d in class_tt.days:
                for ts in class_tt.time_slots:
                    class_tt.timetable[d][ts] = None
        self.timetable.faculty_timetable.clear()
        self.timetable.classroom_timetable.clear()

        env_logger.info("Environment Reset: New training session started.")
        return self.state.flatten(), {}

    def render(self, mode='human'):
        """Render the environment to the screen."""
        env_logger.info("Rendering environment...")
        print("-" * (self.num_classrooms * 15))
        for flat_ts in range(self.num_flat_slots):
            d_i   = flat_ts // self.num_slots_per_day
            s_i   = flat_ts %  self.num_slots_per_day
            day   = self.timetable.days[d_i]
            ts    = self.time_slots[s_i]
            row   = f"{day} {ts}:"
            for room in range(self.num_classrooms):
                ci = self.state[flat_ts, room]
                if ci == 0:
                    row += "  [Empty]    "
                else:
                    course = self.timetable.courses[ci]
                    row += f"  [{course.subject_name}]    "
            print(row)
        print("-" * (self.num_classrooms * 15))

    def seed(self, seed=None):
        """Set the seed for the environment's random number generator."""
        env_logger.info(f"Seeding environment with seed: {seed}")
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed)
        return [seed]
    def get_action_mask(self):
        """
        Returns a 1-D boolean mask of length (n_courses+1 + n_slots + n_rooms),
        where index 0 in the first block is always False (reserved for “no course”),
        and indices 1..n_courses indicate which courses can be scheduled.
        """
        # unpack dims: first dim includes the extra “0” for empty
        n0, n1, n2 = self.action_space.nvec
        # full grid: (courses+1) × slots × rooms
        full = np.zeros((n0, n1, n2), dtype=bool)

        # populate validity (shift course index by +1)
        for ci in range(self.num_courses):
            # skip if course ci already done
            total_done = (self.sessions_scheduled[ci]['theory']
                        + self.sessions_scheduled[ci]['lab'])
            required = self.timetable.courses[ci].credits
            if total_done >= required:
                continue

            for flat_ts in range(self.num_flat_slots):
                day = self.timetable.days[flat_ts // self.num_slots_per_day]
                if day in self.course_day_scheduled[ci]:
                    continue
                slot = self.time_slots[flat_ts % self.num_slots_per_day]

                for rm in range(self.num_classrooms):
                    ok, _ = self.is_valid_action(ci, day, slot, rm)
                    if ok:
                        # mark at index ci+1 so that 0 remains “empty”
                        full[ci + 1, flat_ts, rm] = True

        # reduce into three 1-D masks
        mask_courses = full.any(axis=(1, 2))   # shape (n0,)
        mask_slots   = full.any(axis=(0, 2))   # shape (n1,)
        mask_rooms   = full.any(axis=(0, 1))   # shape (n2,)

        # concatenate into the final mask SB3 expects
        flat_mask = np.concatenate([mask_courses, mask_slots, mask_rooms])
        return flat_mask.astype(bool)

    def find_next_valid(self, course_index, start_flat_ts, classroom):
            """
            Given (course_index, start_flat_ts, classroom), scan forward
            through flat_ts = start_flat_ts, start_flat_ts+1, … up to the very
            last flat index. If you hit the end of the last day, you can wrap
            around to flat_ts = 0 (but usually you only want to scan forward
            within the same episode).
            
            Returns: (new_flat_ts, valid_day, valid_slot_name, valid_reward_term)
            or (None, None, None, penalty) if no valid placement exists anywhere.
            """
            num = self.num_flat_slots
            for delta in range(1, num):  # start search from next slot
                ft = (start_flat_ts + delta) % num
                day_idx  = ft // self.num_slots_per_day
                slot_idx = ft % self.num_slots_per_day
                day      = self.timetable.days[day_idx]
                slot     = self.time_slots[slot_idx]

                # check if course already scheduled on that day → invalid
                if day in self.course_day_scheduled[course_index]:
                    continue

                ok, term = self.is_valid_action(course_index, day, slot, classroom)
                if self.state[ft, classroom] != 0:
                    continue

                if ok:
                    return ft, day, slot, term

            # If we scanned all flat_ts and found none:
            return None, None, None, -20
