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

    # Add time slots first before using them in ClassTimetable
    print("Processing Time Slots...")
    timetable.time_slots = json_data.time_slots  # ✅ Corrected for Pydantic

    # Add faculty members
    print("Processing faculty members...")
    timetable.faculty.extend(
        Faculty(short_name=f.id, full_name=f.name) for f in json_data.faculty  # ✅ Corrected for Pydantic
    )

    # Add classrooms
    print("Processing classrooms...")
    timetable.classrooms.extend(
        Classroom(code=c.id, type=c.type) for c in json_data.classrooms  # ✅ Corrected for Pydantic
    )

    # Process branches and courses
    print("Processing branches and courses...")
    for branch in json_data.branches:  # ✅ Corrected for Pydantic
        print(f"Processing Branch: {branch.branch_name}, Semester: {branch.semester}")

        # Convert courses inside branch
        courses = [
            Course(
                subject_code=course.subject_code,
                subject_name=course.subject_name,
                subject_type=course.subject_type,
                credits=course.credits,
                faculty_id=course.faculty_id
            )
            for course in branch.courses  # ✅ Corrected for Pydantic
        ]

        # Add courses to `timetable.courses` (global list for all branches)
        timetable.courses.extend(courses)  

        # Create a branch object
        branch_obj = Branch(branch_name=branch.branch_name, semester=branch.semester, courses=courses)
        timetable.branches.append(branch_obj)

        # Initialize ClassTimetable object with correct indexing
        branch_sem = f"{branch.branch_name}&{branch.semester}"
        timetable.timetables[branch_sem] = ClassTimetable(branch.branch_name, branch.semester, timetable.time_slots, ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])

    print("Finished processing. Returning Timetable instance.")
    return timetable
