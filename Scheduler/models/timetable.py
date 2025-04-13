from typing import List, Dict, Optional
from models.Faculty import Faculty
from models.Classroom import Classroom
from models.Course import Course
from models.User import User
from datetime import datetime


class TimetableEntry:
    def __init__(self, row: str, col: str, course: Course, faculty: Faculty, classroom: Classroom):
        self.row = row  # Day of the week
        self.col = col  # Time slot
        self.data = {"course": course, "faculty": faculty, "classroom": classroom}


class ClassTimetable:
    def __init__(self, branch: str, semester: str, time_slots: List[str], days: List[str]):
        self.branch = branch
        self.branch_sem = f"{branch}&{semester}"
        self.semester = semester
        self.time_slots = time_slots
        self.days = days
        # Initialize the timetable with None for every slot
        self.timetable = {day: {slot: None for slot in time_slots} for day in days}


class Timetable:
    def __init__(self, id: Optional[str] = None, branches: List[Dict[str, any]] = None):
        self.id = id if id is not None else datetime.now().strftime("%Y%m%d%H%M%S")        
        self.time_slots = ["9AM-10AM", "10AM-11AM", "11AM-12PM", "1PM-2PM", "2PM-3PM"]
        self.days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        self.branches = branches if branches is not None else []
        self.timetables: Dict[str, ClassTimetable] = getattr(self, "timetables", {})

        # Initialize branch-wise timetables only if they are missing
        for b in self.branches:
            branch_sem = f"{b['branch_name']}&{b['semester']}"
            if branch_sem not in self.timetables:
                self.timetables[branch_sem] = ClassTimetable(b["branch_name"], b["semester"], self.time_slots, self.days)

        self.faculty_timetable: Dict[str, Dict[str, Dict[str, Optional[TimetableEntry]]]] = {}
        self.classroom_timetable: Dict[str, Dict[str, Dict[str, Optional[TimetableEntry]]]] = {}
        self.faculty: List[Faculty] = []
        self.courses: List[Course] = []
        self.classrooms: List[Classroom] = []
        self.users: User = User()  # Assuming a single user object for the timetable

    def assign_course(self, branch: str, semester: str, day: str, slot: str, course: Course, faculty: Faculty, classroom: Classroom):
        """Assign a course to a specific time slot and update all relevant timetables."""
        
        branch_sem = f"{branch}&{semester}"  # Create the branch_sem key
        entry = TimetableEntry(day, slot, course, faculty, classroom)

        # Ensure branch_sem exists in timetables
        if branch_sem not in self.timetables:
            self.timetables[branch_sem] = ClassTimetable(branch, semester, self.time_slots, self.days)

        # Update the Student Timetable (ClassTimetable object)
        self.timetables[branch_sem].timetable[day][slot] = entry

        # Update Classroom Timetable
        if classroom.code not in self.classroom_timetable:
            self.classroom_timetable[classroom.code] = {day: {} for day in self.days}
        self.classroom_timetable[classroom.code][day][slot] = entry

        # Update Faculty Timetable
        if faculty.short_name not in self.faculty_timetable:
            self.faculty_timetable[faculty.short_name] = {day: {} for day in self.days}
        self.faculty_timetable[faculty.short_name][day][slot] = entry

    def display(self):
        """Print the timetables for debugging."""
        import json

        def obj_to_dict(obj):
            """Helper function to convert objects to dictionaries for JSON serialization."""
            return obj.__dict__ if not isinstance(obj, str) else obj

        print("Student Timetables by Branch:", json.dumps({k: v.timetable for k, v in self.timetables.items()}, indent=2, default=obj_to_dict))
        print("Classroom Timetable:", json.dumps(self.classroom_timetable, indent=2, default=obj_to_dict))
        print("Faculty Timetable:", json.dumps(self.faculty_timetable, indent=2, default=obj_to_dict))
