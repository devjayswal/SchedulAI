"""
Timetable Environment for PPO Training

Logging Optimization:
- By default, only WARNING and ERROR level logs are written to env.log
- This significantly reduces log file size during training
- To enable verbose logging for debugging, set environment variable:
  TIMETABLE_VERBOSE_LOGGING=true
- Important events (constraint violations, episode completion) are still logged
"""

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


# Configure Logging - Optimized for reduced verbosity
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Environment variable to control logging verbosity
VERBOSE_LOGGING = os.getenv('TIMETABLE_VERBOSE_LOGGING', 'false').lower() == 'true'

env_logger = logging.getLogger("TimetableEnv")
# Set log level based on verbosity setting
env_logger.setLevel(logging.INFO if VERBOSE_LOGGING else logging.WARNING)

# Only add handlers once
if not env_logger.handlers:
    # File handler - only log important events by default
    fh = logging.FileHandler(os.path.join(log_dir, "env.log"))
    fh.setLevel(logging.INFO if VERBOSE_LOGGING else logging.WARNING)

    # Formatter
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)

    env_logger.addHandler(fh)

    # Prevent messages from propagating to the root logger
    env_logger.propagate = False


class TimetableEnv(gym.Env):
    def __init__(self, timetable: Timetable, max_steps=100):
        super().__init__()
        # Only log initialization once per training session
        env_logger.warning("Initializing TimetableEnv...")
        self.timetable = timetable

        # dimensions
        self.time_slots        = [ts for ts in timetable.time_slots if ts != "12:00-13:00"]
        self.num_courses       = len(timetable.courses)
        self.num_days          = len(timetable.days)
        self.num_slots_per_day = len(self.time_slots)
        self.num_classrooms    = len(timetable.classrooms)
        self.num_flat_slots    = self.num_days * self.num_slots_per_day
        self.max_steps         = max_steps

        env_logger.warning(f"Timetable dimensions: {self.num_courses} courses, "
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
        # Removed debug logging to reduce verbosity

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
                # IMPROVED: Reduce logging frequency
                return False, -50

        else:  # lab course
            # lab session?
            is_lab_session = (room_type == "lab")
            if is_lab_session:
                # only 1 lab session
                if self.sessions_scheduled[course_index]['lab'] >= 1:
                    # IMPROVED: Reduce logging frequency
                    return False, -50
                # must fit two consecutive slots
                if slot_idx >= self.num_slots_per_day - 1:
                    # IMPROVED: Reduce logging frequency
                    return False, -50
            else:
                # theory part of a lab course: credits-1 sessions
                if self.sessions_scheduled[course_index]['theory'] >= (credits - 1):
                    # IMPROVED: Reduce logging frequency
                    return False, -50

        # —— 2) Faculty & classroom conflict checks ——
        # single‑slot check
        if (day, time_slot) in self.faculty_schedule[faculty_id]:
            # IMPROVED: Reduce logging frequency
            return False, -50
        if (day, time_slot) in self.classroom_schedule[classroom_obj.code]:
            # IMPROVED: Reduce logging frequency
            return False, -50

        # consecutive faculty‑break check
        if slot_idx > 0:
            prev_slot = self.time_slots[slot_idx - 1]
            if (day, prev_slot) in self.faculty_schedule[faculty_id]:
                # IMPROVED: Reduce logging frequency
                return False, -50

        if slot_idx < self.num_slots_per_day - 1:
            next_slot = self.time_slots[slot_idx + 1]
            if (day, next_slot) in self.faculty_schedule[faculty_id]:
                # IMPROVED: Reduce logging frequency
                return False, -50

        # for lab session, also check the *second* slot
        if ctype == "lab" and room_type == "lab":
            # Check if we can fit two consecutive slots
            if slot_idx >= self.num_slots_per_day - 1:
                # IMPROVED: Don't log every attempt, just return invalid
                return False, -50
            
            second_slot = self.time_slots[slot_idx + 1]
            if (day, second_slot) in self.faculty_schedule[faculty_id] or \
            (day, second_slot) in self.classroom_schedule[classroom_obj.code]:
                # IMPROVED: Don't log every attempt, just return invalid
                return False, -50

        # lunch break
        if time_slot == "12:00-13:00":
            # IMPROVED: Reduce logging frequency
            return False, -50

        # NEW: Course-per-day check
        if day in self.course_day_scheduled[course_index]:
            # IMPROVED: Reduce logging frequency
            return False, -50

        # —— 3) Additional Hard Constraints from SRS ——
        
        # Max 8 classes per day constraint
        daily_classes = sum(1 for slot in self.time_slots 
                           if (day, slot) in self.faculty_schedule[faculty_id] or 
                              (day, slot) in self.classroom_schedule[classroom_obj.code])
        if daily_classes >= 8:
            # IMPROVED: Reduce logging frequency
            return False, -50

        # Non-editable cells constraint (reserved slots)
        reserved_slots = ["12:00-13:00"]  # Lunch break is already handled above
        if time_slot in reserved_slots:
            # IMPROVED: Reduce logging frequency
            return False, -50

        # Faculty consecutive classes in same branch/semester constraint
        # This would require additional tracking of branch/semester per faculty
        # For now, we'll implement a basic version
        if self._has_consecutive_faculty_classes(faculty_id, day, time_slot):
            # IMPROVED: Reduce logging frequency
            return False, -50

        # —— 4) Soft Constraints (penalties) ——
        
        # Edge slot penalty (minimize classes during specific time slots)
        edge_slots = ["09:00-10:00", "16:00-17:00", "17:00-18:00"]
        if time_slot in edge_slots:
            # Removed frequent edge slot penalty logging
            return True, -2

        # Faculty load balancing (soft constraint)
        faculty_load = len(self.faculty_schedule[faculty_id])
        if faculty_load > 6:  # More than 6 classes per week
            # Removed frequent faculty load penalty logging
            return True, -1

        # Removed frequent valid action logging
        return True, 1

    def _has_consecutive_faculty_classes(self, faculty_id, day, time_slot):
        """Check if faculty has consecutive classes in the same branch/semester."""
        slot_idx = self.time_slots.index(time_slot)
        
        # Check previous slot
        if slot_idx > 0:
            prev_slot = self.time_slots[slot_idx - 1]
            if (day, prev_slot) in self.faculty_schedule[faculty_id]:
                return True
        
        # Check next slot
        if slot_idx < self.num_slots_per_day - 1:
            next_slot = self.time_slots[slot_idx + 1]
            if (day, next_slot) in self.faculty_schedule[faculty_id]:
                return True
        
        return False

    def step(self, action):
        """Take a step in the environment.
        action = (course_index, flat_ts, classroom)
        """
        # Removed frequent step logging to reduce verbosity

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
            # Removed frequent slot busy logging
            next_flat, next_day, next_slot, term = self.find_next_valid(ci, flat_ts, classroom)
            if next_flat is None:
                env_logger.warning(f"No valid placement found for course {ci}, ending episode early.")
                return self.state.flatten(), term, True, False, {}  # End episode if no valid placement
            flat_ts, day, time_slot = next_flat, next_day, next_slot
            reward += term

        # 5) Course-per-day check (use ci)
        if day in self.course_day_scheduled[ci]:
            return self.state.flatten(), -50, False, False, {}

        # 6) Validate constraints
        valid, penalty = self.is_valid_action(ci, day, time_slot, classroom)
        if not valid:
            # Check if this course is already fully scheduled
            course = self.timetable.courses[ci]
            required_sessions = course.credits
            completed_sessions = (self.sessions_scheduled[ci]['theory'] + 
                                self.sessions_scheduled[ci]['lab'])
            if completed_sessions >= required_sessions:
                # Removed frequent course completion logging
                return self.state.flatten(), penalty, True, False, {}
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
            # two-slot lab - check if we can fit in current day
            if slot_idx >= self.num_slots_per_day - 1:
                env_logger.warning(f"Lab session cannot fit in last slot of day. Skipping lab scheduling.")
                # Schedule as theory session instead
                self.sessions_scheduled[ci]['theory'] += 1
                class_tt.timetable[day][time_slot] = entry
                self.faculty_schedule[faculty.short_name].add((day, time_slot))
                self.classroom_schedule[classroom_obj.code].add((day, time_slot))
            else:
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

        # 8) Check done - count completed courses instead of non-zero slots
        completed_courses = 0
        for ci in range(self.num_courses):
            course = self.timetable.courses[ci]
            required_sessions = course.credits
            completed_sessions = (self.sessions_scheduled[ci]['theory'] + 
                                self.sessions_scheduled[ci]['lab'])
            if completed_sessions >= required_sessions:
                completed_courses += 1
        
        if completed_courses >= self.num_courses or self.current_step >= self.max_steps:
            done = True
            # Only log episode completion occasionally (every 10th episode)
            if completed_courses % 10 == 0 or completed_courses == self.num_courses:
                env_logger.warning(f"Episode done. Completed {completed_courses}/{self.num_courses} courses.")

        return self.state.flatten(), reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Removed frequent reset logging

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

        # Only log reset occasionally to reduce verbosity
        return self.state.flatten(), {}

    def render(self, mode='human'):
        """Render the environment to the screen."""
        # Removed render logging to reduce verbosity
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
                    course = self.timetable.courses[ci-1]  # ci is shifted index, need to convert back
                    row += f"  [{course.subject_name}]    "
            print(row)
        print("-" * (self.num_classrooms * 15))

    def seed(self, seed=None):
        """Set the seed for the environment's random number generator."""
        # Removed seed logging to reduce verbosity
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed)
        return [seed]
    def get_action_mask(self):
        """
        Returns a 1-D boolean mask of length (n_courses+1 + n_slots + n_rooms),
        where index 0 in the first block is always False (reserved for "no course"),
        and indices 1..n_courses indicate which courses can be scheduled.
        """
        # unpack dims: first dim includes the extra "0" for empty
        n0, n1, n2 = self.action_space.nvec
        # full grid: (courses+1) × slots × rooms
        full = np.zeros((n0, n1, n2), dtype=bool)

        # populate validity (shift course index by +1)
        for ci in range(self.num_courses):
            # skip if course ci already done - IMPROVED CHECK
            total_done = (self.sessions_scheduled[ci]['theory']
                        + self.sessions_scheduled[ci]['lab'])
            required = self.timetable.courses[ci].credits
            if total_done >= required:
                # Mark all actions for this course as invalid
                continue

            # Check if course has any valid placement options
            has_valid_placement = False
            
            for flat_ts in range(self.num_flat_slots):
                day = self.timetable.days[flat_ts // self.num_slots_per_day]
                if day in self.course_day_scheduled[ci]:
                    continue
                slot = self.time_slots[flat_ts % self.num_slots_per_day]

                for rm in range(self.num_classrooms):
                    ok, _ = self.is_valid_action(ci, day, slot, rm)
                    if ok:
                        # mark at index ci+1 so that 0 remains "empty"
                        full[ci + 1, flat_ts, rm] = True
                        has_valid_placement = True

            # If no valid placement found, ensure course is masked out
            if not has_valid_placement:
                # Course has no valid placements, keep it masked
                pass

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
            max_search_attempts = min(num, 50)  # Limit search to prevent infinite loops
            
            for delta in range(1, max_search_attempts):  # start search from next slot
                ft = (start_flat_ts + delta) % num
                day_idx  = ft // self.num_slots_per_day
                slot_idx = ft % self.num_slots_per_day
                day      = self.timetable.days[day_idx]
                slot     = self.time_slots[slot_idx]

                # check if course already scheduled on that day → invalid
                if day in self.course_day_scheduled[course_index]:
                    continue

                # check if slot is already occupied
                if self.state[ft, classroom] != 0:
                    continue

                ok, term = self.is_valid_action(course_index, day, slot, classroom)
                if ok:
                    return ft, day, slot, term

            # If we scanned all flat_ts and found none:
            env_logger.warning(f"No valid placement found for course {course_index} after {max_search_attempts} attempts")
            return None, None, None, -20
