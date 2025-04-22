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
            self.num_courses,
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
                return False, -10

        else:  # lab course
            # lab session?
            is_lab_session = (room_type == "lab")
            if is_lab_session:
                # only 1 lab session
                if self.sessions_scheduled[course_index]['lab'] >= 1:
                    env_logger.warning(f"Invalid action: Lab sessions for course {course_index} "
                                       f"already scheduled.")
                    return False, -10
                # must fit two consecutive slots
                if slot_idx >= self.num_slots_per_day - 1:
                    env_logger.warning("Invalid action: Lab session cannot fit into the last slot.")
                    return False, -10
            else:
                # theory part of a lab course: credits-1 sessions
                if self.sessions_scheduled[course_index]['theory'] >= (credits - 1):
                    env_logger.warning(f"Invalid action: Theory sessions for course {course_index} "
                                       f"already scheduled.")
                    return False, -10

        # —— 2) Faculty & classroom conflict checks ——
        # single‑slot check
        if (day, time_slot) in self.faculty_schedule[faculty_id]:
            env_logger.warning(f"Invalid action: Faculty conflict for {faculty_id} at {day}, {time_slot}.")
            return False, -10
        if (day, time_slot) in self.classroom_schedule[classroom_obj.code]:
            env_logger.warning(f"Invalid action: Classroom conflict for {classroom_obj.code} at {day}, {time_slot}.")

            return False, -10

        # consecutive faculty‑break check

        # consecutive faculty‑break check
        if slot_idx > 0:
            prev_slot = self.time_slots[slot_idx - 1]
            if (day, prev_slot) in self.faculty_schedule[faculty_id]:
                env_logger.warning(f"Faculty has a break just before/after the selected slot.")
                return False, -5

        if slot_idx < self.num_slots_per_day - 1:
            next_slot = self.time_slots[slot_idx + 1]
            if (day, next_slot) in self.faculty_schedule[faculty_id]:
                env_logger.warning(f"Faculty has a break just before/after the selected slot.")
                return False, -5

        # for lab session, also check the *second* slot
        if ctype == "lab" and room_type == "lab":
            # slot_idx < num_slots_per_day - 1 is already guaranteed above
            second_slot = self.time_slots[slot_idx + 1]
            if (day, second_slot) in self.faculty_schedule[faculty_id] or \
            (day, second_slot) in self.classroom_schedule[classroom_obj.code]:
                env_logger.warning(f"Faculty has a break just before/after the selected slot.")
                return False, -10

        # lunch break
        if time_slot == "12:00-13:00":
            env_logger.warning(f"Invalid action: Lunch break at {day}, {time_slot}.")
            return False, -10

        # edge‑slot penalty
        if time_slot in {"09:00-10:00", "15:00-16:00"}:
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
        course_index, flat_ts, classroom = action

        # decode day & slot
        day_idx   = flat_ts // self.num_slots_per_day
        slot_idx  = flat_ts %  self.num_slots_per_day
        day       = self.timetable.days[day_idx]
        time_slot = self.time_slots[slot_idx]

        self.current_step += 1
        reward = 0
        done   = False

        # validate
        valid, penalty = self.is_valid_action(course_index, day, time_slot, classroom)
        if not valid:
            env_logger.warning(f"Invalid action taken: {action}")
            return self.state.flatten(), penalty, False, False, {}
        
        env_logger.info(f"Scheduling course {course_index} at {day}, {time_slot}, classroom {classroom}.")

        # lookup objects
        course        = self.timetable.courses[course_index]
        faculty       = next(f for f in self.timetable.faculty if f.short_name == course.faculty_id)
        classroom_obj = self.timetable.classrooms[classroom]

        # identify branch‑semester
        branch_sem = next(
            f"{b.branch_name}&{b.semester}"
            for b in self.timetable.branches
            if course in b.courses
        )
        class_tt = self.timetable.timetables[branch_sem]

        # determine lab vs theory session
        is_lab_session = (course.subject_type == "lab" and classroom_obj.type == "lab")

        # create entry
        entry = TimetableEntry(day, time_slot, course, faculty, classroom_obj)

        # ——— schedule it ——
        if is_lab_session:
            # occupy two slots
            next_flat_ts = flat_ts + 1
            next_slot    = self.timetable.time_slots[slot_idx + 1]

            # branch timetable
            class_tt.timetable[day][time_slot] = entry
            class_tt.timetable[day][next_slot] = entry

            # faculty timetable
            self.timetable.faculty_timetable.setdefault(faculty.short_name, {})\
                                            .setdefault(day, {})[time_slot] = entry
            self.timetable.faculty_timetable[faculty.short_name][day][next_slot] = entry

            # classroom timetable
            self.timetable.classroom_timetable.setdefault(classroom_obj.code, {})\
                                               .setdefault(day, {})[time_slot] = entry
            self.timetable.classroom_timetable[classroom_obj.code][day][next_slot] = entry

            # RL state
            self.state[flat_ts, classroom]      = course_index
            self.state[next_flat_ts, classroom] = course_index

            # conflict trackers
            self.faculty_schedule[faculty.short_name].update({(day, time_slot), (day, next_slot)})
            self.classroom_schedule[classroom_obj.code].update({(day, time_slot), (day, next_slot)})

            # count it
            self.sessions_scheduled[course_index]['lab'] += 1

        else:
            # single‑slot theory
            class_tt.timetable[day][time_slot] = entry
            self.timetable.faculty_timetable.setdefault(faculty.short_name, {})\
                                            .setdefault(day, {})[time_slot] = entry
            self.timetable.classroom_timetable.setdefault(classroom_obj.code, {})\
                                               .setdefault(day, {})[time_slot] = entry

            self.state[flat_ts, classroom] = course_index
            self.faculty_schedule[faculty.short_name].add((day, time_slot))
            self.classroom_schedule[classroom_obj.code].add((day, time_slot))

            # count it
            self.sessions_scheduled[course_index]['theory'] += 1

        # reward & done
        reward += penalty
        if (np.count_nonzero(self.state) == self.num_courses or
            self.current_step >= self.max_steps):
            done = True
            env_logger.info("All courses scheduled or max steps reached. Ending episode.")
        return self.state.flatten(), reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        env_logger.info("Resetting environment...")

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
        np.random.seed(seed)
        random.seed(seed)
        return [seed]
    

    def get_valid_actions(self):
        valid = []
        for ci in range(self.num_courses):
            for flat_ts in range(self.num_flat_slots):
                day = self.timetable.days[flat_ts // self.num_slots_per_day]
                ts  = self.time_slots[flat_ts % self.num_slots_per_day]
                for room in range(self.num_classrooms):
                    ok,_ = self.is_valid_action(ci, day, ts, room)
                    if ok:
                        valid.append((ci, flat_ts, room))
        return valid

    def sample_valid_action(self):
        return random.choice(self.get_valid_actions())
    
    def get_action_mask(self):
        n0, n1, n2 = self.action_space.nvec

        full = np.zeros((n0, n1, n2), dtype=bool)
        for ci, flat_ts, rm in self.get_valid_actions():
            full[ci, flat_ts, rm] = True

        mask0 = full.any(axis=(1, 2))  # shape (n0,)
        mask1 = full.any(axis=(0, 2))  # shape (n1,)
        mask2 = full.any(axis=(0, 1))  # shape (n2,)

        return np.concatenate([mask0, mask1, mask2])  # list of bool arrays (one per action dim)
