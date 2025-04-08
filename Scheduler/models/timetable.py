from typing import List, Dict, Optional
from models import User, Course, Faculty, Classroom  # Import models properly


class TimetableEntry:
    def __init__(self, row: str, col: str, course: Course, faculty: Faculty, classroom: Classroom):
        self.row = row  # Day of the week
        self.col = col  # Time slot
        self.data = {"course": course, "faculty": faculty, "classroom": classroom}


class Timetable:
    def __init__(self, id: Optional[str] = None, branch: str = "", sem: int = 0):
        self.id = id
        self.branch = branch
        self.sem = sem
        self.timetables: Dict[str, Dict[str, Optional[TimetableEntry]]] = {}
        self.faculty_timetable: Dict[str, Dict[str, Optional[TimetableEntry]]] = {}
        self.classroom_timetable: Dict[str, Dict[str, Optional[TimetableEntry]]] = {}
        self.faculty: List[Faculty] = []
        self.courses: List[Course] = []
        self.classrooms: List[Classroom] = []
        # user is single user object, not a list of users
        self.users: User = User()  # Assuming a single user object for the timetable

    def _init_timetable(self):
        """Initialize an empty timetable with days and slots."""
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        slots = ["9AM-10AM", "10AM-11AM", "11AM-12PM", "1PM-2PM", "2PM-3PM"]
        return {day: {slot: None for slot in slots} for day in days}

    def assign_course(self, branch: str, day: str, slot: str, course: Course, faculty: Faculty, classroom: Classroom):
        """Assign a course to a specific time slot and update all relevant timetables."""
        entry = TimetableEntry(day, slot, course, faculty, classroom)

        # Update Student Timetable (categorized by branch)
        if branch not in self.timetables:
            self.timetables[branch] = self._init_timetable()
        self.timetables[branch][day][slot] = entry

        # Update Classroom Timetable
        if day not in self.classroom_timetable:
            self.classroom_timetable[day] = {}
        self.classroom_timetable[day][slot] = entry

        # Update Faculty Timetable
        if day not in self.faculty_timetable:
            self.faculty_timetable[day] = {}
        self.faculty_timetable[day][slot] = entry

    def display(self):
        """Print the timetables for debugging."""
        import json

        def obj_to_dict(obj):
            """Helper function to convert objects to dictionaries for JSON serialization."""
            return obj.__dict__ if not isinstance(obj, str) else obj

        print("Student Timetables by Branch:", json.dumps(self.timetables, indent=2, default=obj_to_dict))
        print("Classroom Timetable:", json.dumps(self.classroom_timetable, indent=2, default=obj_to_dict))
        print("Faculty Timetable:", json.dumps(self.faculty_timetable, indent=2, default=obj_to_dict))
