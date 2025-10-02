from models.Timetable import Timetable, ClassTimetable  # Import ClassTimetable
import uuid
from models.Course import Course
from models.Faculty import Faculty
from models.Classroom import Classroom
from models.Branch import Branch

def jsonToClass(json_data):
    print("Starting jsonToClass function...")

    # Create Timetable instance
    timetable = Timetable(id=str(uuid.uuid4()))
    print(f"Created Timetable instance with ID: {timetable.id}")

    # Handle both dictionary and Pydantic model inputs
    def get_value(data, key):
        """Get value from either dict or Pydantic model."""
        if isinstance(data, dict):
            return data.get(key)
        else:
            return getattr(data, key, None)

    def get_attr(obj, attr):
        """Get attribute from either dict or Pydantic model."""
        if isinstance(obj, dict):
            return obj.get(attr)
        else:
            return getattr(obj, attr, None)

    # Add time slots first before using them in ClassTimetable
    print("Processing Time Slots...")
    timetable.time_slots = get_value(json_data, "time_slots") or []

    # Add faculty members
    print("Processing faculty members...")
    faculty_data = get_value(json_data, "faculty") or []
    timetable.faculty.extend(
        Faculty(short_name=get_attr(f, "id"), full_name=get_attr(f, "name")) for f in faculty_data
    )

    # Add classrooms
    print("Processing classrooms...")
    classroom_data = get_value(json_data, "classrooms") or []
    timetable.classrooms.extend(
        Classroom(code=get_attr(c, "id"), type=get_attr(c, "type")) for c in classroom_data
    )

    # Process branches and courses
    print("Processing branches and courses...")
    branches_data = get_value(json_data, "branches") or []
    for branch in branches_data:
        branch_name = get_attr(branch, "branch_name")
        branch_semester = get_attr(branch, "semester")
        print(f"Processing Branch: {branch_name}, Semester: {branch_semester}")

        # Convert courses inside branch
        courses_data = get_attr(branch, "courses") or []
        courses = [
            Course(
                subject_code=get_attr(course, "subject_code"),
                subject_name=get_attr(course, "subject_name"),
                subject_type=get_attr(course, "subject_type"),
                credits=get_attr(course, "credits"),
                faculty_id=get_attr(course, "faculty_id")
            )
            for course in courses_data
        ]

        # Add courses to `timetable.courses` (global list for all branches)
        timetable.courses.extend(courses)  

        # Create a branch object
        branch_obj = Branch(branch_name=branch_name, semester=branch_semester, courses=courses)
        timetable.branches.append(branch_obj)

        # Initialize ClassTimetable object with correct indexing
        branch_sem = f"{branch_name}&{branch_semester}"
        timetable.timetables[branch_sem] = ClassTimetable(branch_name, branch_semester, timetable.time_slots, ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])

    print("Finished processing. Returning Timetable instance.")
    return timetable
