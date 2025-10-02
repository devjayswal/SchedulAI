"""
Improved Timetable Environment for PPO Training
This version addresses the key issues identified in the training analysis:
1. Better reward shaping with positive guidance
2. Improved action masking
3. Curriculum learning support
4. More informative state representation
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


# Configure Logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

VERBOSE_LOGGING = os.getenv('TIMETABLE_VERBOSE_LOGGING', 'false').lower() == 'true'

env_logger = logging.getLogger("TimetableEnv")
env_logger.setLevel(logging.INFO if VERBOSE_LOGGING else logging.WARNING)

if not env_logger.handlers:
    fh = logging.FileHandler(os.path.join(log_dir, "enhanced_env.log"))
    fh.setLevel(logging.INFO if VERBOSE_LOGGING else logging.WARNING)
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    env_logger.addHandler(fh)
    env_logger.propagate = False


class TimetableEnv(gym.Env):
    def __init__(self, timetable: Timetable, max_steps=100, difficulty_level=0):
        super().__init__()
        env_logger.warning("Initializing TimetableEnv...")
        self.timetable = timetable
        self.difficulty_level = difficulty_level  # 0=easy, 1=medium, 2=hard

        # dimensions
        self.time_slots = [ts for ts in timetable.time_slots if ts != "12:00-13:00"]
        self.num_courses = len(timetable.courses)
        self.num_days = len(timetable.days)
        self.num_slots_per_day = len(self.time_slots)
        self.num_classrooms = len(timetable.classrooms)
        self.num_flat_slots = self.num_days * self.num_slots_per_day
        self.max_steps = max_steps

        env_logger.warning(f"Timetable dimensions: {self.num_courses} courses, "
                        f"{self.num_days} days, {self.num_slots_per_day} slots/day, "
                        f"{self.num_classrooms} classrooms, difficulty: {difficulty_level}")

        # action & observation spaces
        self.action_space = MultiDiscrete((
            self.num_courses + 1,  # +1 for "no action"
            self.num_flat_slots,
            self.num_classrooms,
        ))
        
        # Enhanced observation space with more information
        obs_size = (self.num_flat_slots * self.num_classrooms +  # current state
                   self.num_courses +  # course completion status
                   self.num_courses +  # course session counts
                   self.num_flat_slots +  # slot availability
                   self.num_classrooms)  # classroom availability
        
        self.observation_space = Box(
            low=0,
            high=1,
            shape=(obs_size,),
            dtype=np.float32,
        )

        # conflict trackers
        self.faculty_schedule = defaultdict(set)
        self.classroom_schedule = defaultdict(set)
        self.course_day_scheduled = {i: set() for i in range(self.num_courses)}
        self.sessions_scheduled = defaultdict(lambda: {'theory': 0, 'lab': 0})

        # Performance tracking
        self.episode_rewards = []
        self.episode_length = 0
        self.successful_placements = 0
        self.constraint_violations = 0

        self.reset()

    def get_enhanced_observation(self):
        """Get enhanced observation with more state information."""
        obs = []
        
        # Current state (flattened)
        obs.extend(self.state.flatten().astype(np.float32))
        
        # Course completion status (0=not started, 0.5=partial, 1=complete)
        course_status = []
        for i in range(self.num_courses):
            course = self.timetable.courses[i]
            required = course.credits
            completed = (self.sessions_scheduled[i]['theory'] + 
                        self.sessions_scheduled[i]['lab'])
            if completed == 0:
                course_status.append(0.0)
            elif completed >= required:
                course_status.append(1.0)
            else:
                course_status.append(0.5)
        obs.extend(course_status)
        
        # Course session counts (normalized)
        session_counts = []
        for i in range(self.num_courses):
            course = self.timetable.courses[i]
            required = course.credits
            completed = (self.sessions_scheduled[i]['theory'] + 
                        self.sessions_scheduled[i]['lab'])
            session_counts.append(completed / max(required, 1))
        obs.extend(session_counts)
        
        # Slot availability (0=occupied, 1=available)
        slot_availability = []
        for flat_ts in range(self.num_flat_slots):
            if np.any(self.state[flat_ts, :] == 0):
                slot_availability.append(1.0)
            else:
                slot_availability.append(0.0)
        obs.extend(slot_availability)
        
        # Classroom availability (0=occupied, 1=available)
        classroom_availability = []
        for classroom in range(self.num_classrooms):
            if np.any(self.state[:, classroom] == 0):
                classroom_availability.append(1.0)
            else:
                classroom_availability.append(0.0)
        obs.extend(classroom_availability)
        
        return np.array(obs, dtype=np.float32)

    def is_valid_action(self, course_index, day, time_slot, classroom):
        """Improved constraint validation with better feedback."""
        course = self.timetable.courses[course_index]
        credits = course.credits
        ctype = course.subject_type
        faculty_id = course.faculty_id
        classroom_obj = self.timetable.classrooms[classroom]
        room_type = classroom_obj.type
        slot_idx = self.time_slots.index(time_slot)

        # 1) Session count constraints
        if ctype == "theory":
            if self.sessions_scheduled[course_index]['theory'] >= credits:
                return False, -10  # Reduced penalty
        else:  # lab course
            is_lab_session = (room_type == "lab")
            if is_lab_session:
                if self.sessions_scheduled[course_index]['lab'] >= 1:
                    return False, -10
                if slot_idx >= self.num_slots_per_day - 1:
                    return False, -10
            else:
                if self.sessions_scheduled[course_index]['theory'] >= (credits - 1):
                    return False, -10

        # 2) Faculty & classroom conflicts
        if (day, time_slot) in self.faculty_schedule[faculty_id]:
            return False, -15  # Reduced penalty
        if (day, time_slot) in self.classroom_schedule[classroom_obj.code]:
            return False, -15

        # 3) Faculty break constraints (relaxed for easier learning)
        if self.difficulty_level >= 1:  # Only enforce in medium+ difficulty
            if slot_idx > 0:
                prev_slot = self.time_slots[slot_idx - 1]
                if (day, prev_slot) in self.faculty_schedule[faculty_id]:
                    return False, -5

            if slot_idx < self.num_slots_per_day - 1:
                next_slot = self.time_slots[slot_idx + 1]
                if (day, next_slot) in self.faculty_schedule[faculty_id]:
                    return False, -5

        # 4) Lab session constraints
        if ctype == "lab" and room_type == "lab":
            if slot_idx >= self.num_slots_per_day - 1:
                return False, -10
            
            second_slot = self.time_slots[slot_idx + 1]
            if (day, second_slot) in self.faculty_schedule[faculty_id] or \
               (day, second_slot) in self.classroom_schedule[classroom_obj.code]:
                return False, -10

        # 5) Course-per-day constraint (relaxed for easier learning)
        if self.difficulty_level >= 2:  # Only enforce in hard difficulty
            if day in self.course_day_scheduled[course_index]:
                return False, -5

        # 6) Lunch break
        if time_slot == "12:00-13:00":
            return False, -20

        # 7) Daily class limit (relaxed)
        if self.difficulty_level >= 2:
            daily_classes = sum(1 for slot in self.time_slots 
                              if (day, slot) in self.faculty_schedule[faculty_id])
            if daily_classes >= 8:
                return False, -5

        # Soft constraints (penalties, not hard blocks)
        penalty = 0
        
        # Edge slot penalty (reduced)
        edge_slots = ["09:00-10:00", "16:00-17:00", "17:00-18:00"]
        if time_slot in edge_slots:
            penalty -= 1

        # Faculty load balancing (reduced)
        faculty_load = len(self.faculty_schedule[faculty_id])
        if faculty_load > 6:
            penalty -= 1

        return True, 5 + penalty  # Positive base reward for valid actions

    def step(self, action):
        """Improved step function with better reward shaping."""
        course_index, flat_ts, classroom = action
        course_index = int(course_index)
        flat_ts = int(flat_ts)
        classroom = int(classroom)

        self.episode_length += 1
        reward = 0
        done = False
        info = {}

        # Handle "no action"
        if course_index == 0:
            reward = -1  # Small penalty for inaction
            return self.get_enhanced_observation(), reward, done, False, info

        # Map to real course index
        ci = course_index - 1

        # Validate indices
        if not (0 <= ci < self.num_courses):
            reward = -20
            return self.get_enhanced_observation(), reward, done, False, info

        # Decode day & slot
        day_idx = flat_ts // self.num_slots_per_day
        slot_idx = flat_ts % self.num_slots_per_day
        day = self.timetable.days[day_idx]
        time_slot = self.time_slots[slot_idx]

        # Check if slot is occupied
        if self.state[flat_ts, classroom] != 0:
            # Try to find next valid slot
            next_flat, next_day, next_slot, term = self.find_next_valid(ci, flat_ts, classroom)
            if next_flat is None:
                reward = -5  # Reduced penalty for no valid placement
                return self.get_enhanced_observation(), reward, done, False, info
            flat_ts, day, time_slot = next_flat, next_day, next_slot
            reward += term

        # Validate constraints
        valid, penalty = self.is_valid_action(ci, day, time_slot, classroom)
        if not valid:
            self.constraint_violations += 1
            # Check if course is already complete
            course = self.timetable.courses[ci]
            required_sessions = course.credits
            completed_sessions = (self.sessions_scheduled[ci]['theory'] + 
                                self.sessions_scheduled[ci]['lab'])
            if completed_sessions >= required_sessions:
                reward = 10  # Positive reward for trying to schedule completed course
                return self.get_enhanced_observation(), reward, done, False, info
            return self.get_enhanced_observation(), penalty, done, False, info

        # Successful placement
        self.successful_placements += 1
        reward += penalty

        # Perform scheduling
        course = self.timetable.courses[ci]
        faculty = next(f for f in self.timetable.faculty if f.short_name == course.faculty_id)
        classroom_obj = self.timetable.classrooms[classroom]

        # Branch timetable lookup
        branch_sem = next(
            f"{b.branch_name}&{b.semester}"
            for b in self.timetable.branches
            if course in b.courses
        )
        class_tt = self.timetable.timetables[branch_sem]

        # Create entry and update state
        entry = TimetableEntry(day, time_slot, course, faculty, classroom_obj)
        self.state[flat_ts, classroom] = course_index

        # Update trackers
        self.course_day_scheduled[ci].add(day)
        
        if course.subject_type == "lab" and classroom_obj.type == "lab":
            # Two-slot lab
            if slot_idx >= self.num_slots_per_day - 1:
                # Schedule as theory session instead
                self.sessions_scheduled[ci]['theory'] += 1
                class_tt.timetable[day][time_slot] = entry
                self.faculty_schedule[faculty.short_name].add((day, time_slot))
                self.classroom_schedule[classroom_obj.code].add((day, time_slot))
                reward += 5  # Bonus for successful adaptation
            else:
                # Two-slot lab
                next_flat_ts = flat_ts + 1
                next_slot = self.time_slots[slot_idx + 1]
                self.state[next_flat_ts, classroom] = course_index
                self.sessions_scheduled[ci]['lab'] += 1
                class_tt.timetable[day][time_slot] = entry
                class_tt.timetable[day][next_slot] = entry
                self.faculty_schedule[faculty.short_name].update({(day, time_slot), (day, next_slot)})
                self.classroom_schedule[classroom_obj.code].update({(day, time_slot), (day, next_slot)})
                reward += 10  # Bonus for successful lab scheduling
        else:
            # Single-slot theory
            self.sessions_scheduled[ci]['theory'] += 1
            class_tt.timetable[day][time_slot] = entry
            self.faculty_schedule[faculty.short_name].add((day, time_slot))
            self.classroom_schedule[classroom_obj.code].add((day, time_slot))
            reward += 5  # Bonus for successful theory scheduling

        # Check completion
        completed_courses = 0
        for ci in range(self.num_courses):
            course = self.timetable.courses[ci]
            required_sessions = course.credits
            completed_sessions = (self.sessions_scheduled[ci]['theory'] + 
                                self.sessions_scheduled[ci]['lab'])
            if completed_sessions >= required_sessions:
                completed_courses += 1

        # Episode completion
        if completed_courses >= self.num_courses:
            done = True
            reward += 100  # Large bonus for completing all courses
            info['episode_completed'] = True
            info['success'] = True
        elif self.episode_length >= self.max_steps:
            done = True
            # Partial completion bonus
            completion_rate = completed_courses / self.num_courses
            reward += completion_rate * 50
            info['episode_completed'] = False
            info['success'] = False

        # Additional info
        info.update({
            'completed_courses': completed_courses,
            'total_courses': self.num_courses,
            'successful_placements': self.successful_placements,
            'constraint_violations': self.constraint_violations,
            'utilization_rate': self.calculate_utilization_rate(),
            'episode_length': self.episode_length
        })

        self.episode_rewards.append(reward)
        return self.get_enhanced_observation(), reward, done, False, info

    def calculate_utilization_rate(self):
        """Calculate resource utilization rate."""
        total_slots = self.num_flat_slots * self.num_classrooms
        occupied_slots = np.sum(self.state != 0)
        return occupied_slots / total_slots if total_slots > 0 else 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset tracking variables
        for i in range(self.num_courses):
            self.course_day_scheduled[i].clear()

        # Reset state
        self.state = np.zeros((self.num_flat_slots, self.num_classrooms), dtype=np.int32)
        self.episode_length = 0
        self.successful_placements = 0
        self.constraint_violations = 0
        self.episode_rewards = []

        # Reset trackers
        self.faculty_schedule.clear()
        self.classroom_schedule.clear()
        self.sessions_scheduled.clear()

        # Clear underlying timetable
        for class_tt in self.timetable.timetables.values():
            for d in class_tt.days:
                for ts in class_tt.time_slots:
                    class_tt.timetable[d][ts] = None
        self.timetable.faculty_timetable.clear()
        self.timetable.classroom_timetable.clear()

        return self.get_enhanced_observation(), {}

    def get_action_mask(self):
        """Improved action masking with better efficiency."""
        n0, n1, n2 = self.action_space.nvec
        full = np.zeros((n0, n1, n2), dtype=bool)

        # Mark "no action" as always valid
        full[0, :, :] = True

        # Check valid actions for each course
        for ci in range(self.num_courses):
            # Skip if course is already complete
            total_done = (self.sessions_scheduled[ci]['theory'] +
                        self.sessions_scheduled[ci]['lab'])
            required = self.timetable.courses[ci].credits
            if total_done >= required:
                continue

            # Check valid placements
            for flat_ts in range(self.num_flat_slots):
                day = self.timetable.days[flat_ts // self.num_slots_per_day]
                
                # Skip if course already scheduled on this day (for higher difficulty)
                if self.difficulty_level >= 2 and day in self.course_day_scheduled[ci]:
                    continue
                    
                slot = self.time_slots[flat_ts % self.num_slots_per_day]

                for rm in range(self.num_classrooms):
                    # Skip if slot is occupied
                    if self.state[flat_ts, rm] != 0:
                        continue
                        
                    ok, _ = self.is_valid_action(ci, day, slot, rm)
                    if ok:
                        full[ci + 1, flat_ts, rm] = True

        # Create masks
        mask_courses = full.any(axis=(1, 2))
        mask_slots = full.any(axis=(0, 2))
        mask_rooms = full.any(axis=(0, 1))

        flat_mask = np.concatenate([mask_courses, mask_slots, mask_rooms])
        return flat_mask.astype(bool)

    def find_next_valid(self, course_index, start_flat_ts, classroom):
        """Find next valid placement for a course."""
        num = self.num_flat_slots
        max_search_attempts = min(num, 20)  # Reduced search space
        
        for delta in range(1, max_search_attempts):
            ft = (start_flat_ts + delta) % num
            day_idx = ft // self.num_slots_per_day
            slot_idx = ft % self.num_slots_per_day
            day = self.timetable.days[day_idx]
            slot = self.time_slots[slot_idx]

            # Check constraints
            if day in self.course_day_scheduled[course_index] and self.difficulty_level >= 2:
                continue
            if self.state[ft, classroom] != 0:
                continue

            ok, term = self.is_valid_action(course_index, day, slot, classroom)
            if ok:
                return ft, day, slot, term

        return None, None, None, -5

    def set_difficulty(self, level):
        """Set difficulty level for curriculum learning."""
        self.difficulty_level = max(0, min(2, level))
        env_logger.info(f"Difficulty level set to {self.difficulty_level}")

    def get_performance_metrics(self):
        """Get current episode performance metrics."""
        return {
            'episode_length': self.episode_length,
            'successful_placements': self.successful_placements,
            'constraint_violations': self.constraint_violations,
            'utilization_rate': self.calculate_utilization_rate(),
            'total_reward': sum(self.episode_rewards),
            'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0
        }

    def render(self, mode='human'):
        """Render the environment."""
        print("-" * (self.num_classrooms * 15))
        for flat_ts in range(self.num_flat_slots):
            d_i = flat_ts // self.num_slots_per_day
            s_i = flat_ts % self.num_slots_per_day
            day = self.timetable.days[d_i]
            ts = self.time_slots[s_i]
            row = f"{day} {ts}:"
            for room in range(self.num_classrooms):
                ci = self.state[flat_ts, room]
                if ci == 0:
                    row += "  [Empty]    "
                else:
                    course = self.timetable.courses[ci-1]
                    row += f"  [{course.subject_name}]    "
            print(row)
        print("-" * (self.num_classrooms * 15))
        
        # Print metrics
        metrics = self.get_performance_metrics()
        print(f"Metrics: {metrics}")

    def seed(self, seed=None):
        """Set the seed for the environment's random number generator."""
        self.np_random, seed = seeding.np_random(seed)
        random.seed(seed)
        return [seed]
