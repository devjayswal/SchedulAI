import numpy as np
import gym
from gym import spaces
from collections import defaultdict
from domain import Course, Faculty, Classroom
from models import Timetable, TimetableEntry

class TimetableEnv(gym.Env):
    def __init__(self, timetable: Timetable, max_steps=100):
        super(TimetableEnv, self).__init__()

        self.timetable = timetable  # Timetable object to store results
        self.num_courses = len(timetable.courses)
        self.num_slots = len(timetable.timetables[next(iter(timetable.timetables))])  # Assuming all branches have same slots
        self.num_classrooms = len(timetable.classrooms)
        self.max_steps = max_steps

        # Define action space: (course_index, time_slot, classroom)
        self.action_space = spaces.MultiDiscrete([self.num_courses, self.num_slots, self.num_classrooms])
        
        # Define observation space
        self.observation_space = spaces.Box(low=0, high=self.num_courses, shape=(self.num_slots * self.num_classrooms,), dtype=np.int32)

        # Tracking schedules
        self.faculty_schedule = defaultdict(set)  # {faculty_id: {time_slots}}
        self.classroom_schedule = defaultdict(set)  # {classroom_id: {time_slots}}

        self.reset()

    def is_valid_action(self, course_index, time_slot, classroom):
        """ Checks all constraints before scheduling a course """
        course = self.timetable.courses[course_index]
        faculty_id = course.faculty_id
        classroom_id = self.timetable.classrooms[classroom].id

        # HARD CONSTRAINTS:
        # 1. Avoid scheduling two classes for the same faculty at the same time
        if time_slot in self.faculty_schedule[faculty_id]:
            return False, -10  # Conflict penalty

        # 2. Avoid consecutive classes for the same faculty
        if (time_slot - 1 in self.faculty_schedule[faculty_id]) or (time_slot + 1 in self.faculty_schedule[faculty_id]):
            return False, -5  # Consecutive class penalty

        # 3. Avoid double-booking a classroom
        if time_slot in self.classroom_schedule[classroom_id]:
            return False, -10

        # 4. Reserve 1:00-2:00 PM for lunch (assuming 13:00-14:00 is slot index 4)
        if time_slot == 4:
            return False, -10

        # SOFT CONSTRAINTS:
        # 5. Minimize classes in specific slots (09:00-10:00, 16:00-17:00, 17:00-18:00)
        if time_slot in {0, 8, 9}:  # Assuming these are the indices
            return True, -2  # Small penalty

        return True, 1  # Valid action with reward

    def step(self, action):
        course_index, time_slot, classroom = action
        reward = 0
        done = False

        self.current_step += 1

        # Validate the action
        valid, penalty = self.is_valid_action(course_index, time_slot, classroom)
        if not valid:
            return self.state.flatten(), penalty, False, {}  # Return penalty for invalid move

        # Assign course to timetable
        course = self.timetable.courses[course_index]
        faculty = next(f for f in self.timetable.faculty if f.id == course.faculty_id)
        classroom_obj = self.timetable.classrooms[classroom]

        # Find day and slot based on index
        days = list(self.timetable.timetables[next(iter(self.timetable.timetables))].keys())  # Get weekdays
        slots = list(self.timetable.timetables[next(iter(self.timetable.timetables))][days[0]].keys())  # Get time slots
        day = days[time_slot // len(slots)]  # Determine day
        slot = slots[time_slot % len(slots)]  # Determine time slot

        # Create timetable entry
        entry = TimetableEntry(day, slot, course, faculty, classroom_obj)

        # Update Student Timetable
        branch = self.timetable.branch
        self.timetable.timetables[branch][day][slot] = entry

        # Update Faculty Timetable
        self.timetable.faculty_timetable[day][slot] = entry

        # Update Classroom Timetable
        self.timetable.classroom_timetable[day][slot] = entry

        # Update state tracking
        self.state[time_slot, classroom] = course_index
        self.faculty_schedule[faculty.id].add(time_slot)
        self.classroom_schedule[classroom_obj.id].add(time_slot)

        reward += penalty  # Add soft constraint reward/penalty

        # Check if scheduling is complete
        if np.count_nonzero(self.state) == self.num_courses or self.current_step >= self.max_steps:
            done = True

        return self.state.flatten(), reward, done, {}

    def reset(self):
        self.state = np.zeros((self.num_slots, self.num_classrooms), dtype=np.int32)
        self.current_step = 0
        self.faculty_schedule.clear()
        self.classroom_schedule.clear()
        return self.timetable  # Return the updated timetable object

    def render(self, mode='human'):
        """ Prints the timetable in a structured format """
        print("\nTimetable:")
        print("-" * (self.num_classrooms * 15))
        
        for slot in range(self.num_slots):
            row = f"Slot {slot+1}:"
            for room in range(self.num_classrooms):
                course_index = self.state[slot, room]
                if course_index == 0:
                    row += "  [Empty]    "
                else:
                    course = self.timetable.courses[course_index]
                    row += f"  [{course.name}]    "
            print(row)
        
        print("-" * (self.num_classrooms * 15))
        print("Faculty Schedule:")
        for faculty_id, slots in self.faculty_schedule.items():
            print(f"Faculty {faculty_id}: {slots}")
        print("Classroom Schedule:")
        for classroom_id, slots in self.classroom_schedule.items():
            print(f"Classroom {classroom_id}: {slots}")
        print("-" * (self.num_classrooms * 15))