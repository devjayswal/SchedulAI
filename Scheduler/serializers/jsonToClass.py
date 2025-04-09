from models import User, Course, Faculty, Classroom
from timetable import Timetable  # Import your Timetable class
import uuid

def jsonToClass(json_data):
    print("Starting jsonToClass function...")

    # Create Timetable instance
    timetable = Timetable(
        id=str(uuid.uuid4()),
    )
    print(f"Created Timetable instance with ID: {timetable.id}")

    # Add faculty members
    faculty_dict = {}
    print("Processing faculty members...")
    for f in json_data["faculty"]:
        faculty_obj = Faculty(id=f["id"], name=f["name"], email=f["email"])
        faculty_dict[f["id"]] = faculty_obj
        timetable.faculty.append(faculty_obj)
        print(f"Added Faculty: {faculty_obj}")

    # Add classrooms
    classroom_dict = {}
    print("Processing classrooms...")
    for c in json_data["classrooms"]:
        classroom_obj = Classroom(id=c["id"], type=c["type"])
        classroom_dict[c["id"]] = classroom_obj
        timetable.classrooms.append(classroom_obj)
        print(f"Added Classroom: {classroom_obj}")

    # Process branches and courses
    print("Processing branches and courses...")
    for branch in json_data["branches"]:
        branch_name = branch["branch_name"]
        sem = branch["semester"]
        timetable.branch = branch_name
        timetable.sem = sem
        print(f"Processing Branch: {branch_name}, Semester: {sem}")
        
        # Initialize timetable structure
        if branch_name not in timetable.timetables:
            timetable.timetables[branch_name] = timetable._init_timetable()
            print(f"Initialized timetable structure for branch: {branch_name}")

        for course in branch["courses"]:
            # Create Course object
            course_obj = Course(
                subject_code=course["subject_code"],
                subject_name=course["subject_name"],
                subject_type=course["subject_type"],
                credits=course["credits"],
                faculty_id=course["faculty_id"]
            )
            timetable.courses.append(course_obj)
            print(f"Added Course: {course_obj}")

    print("Finished processing. Returning Timetable instance.")
    return timetable