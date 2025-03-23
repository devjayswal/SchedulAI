import json
import numpy as np
import os

# Create resources directory if it doesn't exist
RESOURCE_DIR = "resources"
os.makedirs(RESOURCE_DIR, exist_ok=True)

def create_rooms_json(filename="rooms.json"):
    # Create dummy room data: 10 rooms, with room_no and a flag indicating if it's a lab.
    rooms = []
    for i in range(10):
        room = {
            "room_no": str(101 + i),
            "is_lab": True if i < 4 else False  # first 4 are labs
        }
        rooms.append(room)
    data = {"rooms": rooms}
    
    filepath = os.path.join(RESOURCE_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Rooms JSON saved to {filepath}")

def create_faculty_json(filename="faculty.json"):
    # Create a list of faculty records.
    faculties = []
    for fid in range(1, 11):
        faculties.append({
            "faculty_id": f"FAC{fid:02d}",
            "name": f"Faculty_{fid}"
        })
    data = {"faculty": faculties}
    
    filepath = os.path.join(RESOURCE_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Faculty JSON saved to {filepath}")
    return [f["faculty_id"] for f in faculties]  # Return faculty IDs for subject assignment

def create_subjects_json(faculty_ids, filename="subjects.json"):
    # There are 15 classes (3 sems * 5 branches) with 5 subjects each (75 subjects total)
    subjects = []
    branches = ["AIR", "AIML", "AIDS", "AI", "CSE"]
    sems = [2, 4, 6]
    subject_id = 1
    for sem in sems:
        for branch in branches:
            for _ in range(5):
                credits = np.random.choice([2, 3])
                subject_type = "mixed" if credits == 3 else np.random.choice(["theory", "practical"])
                faculty_code = np.random.choice(faculty_ids)  # Assign a faculty randomly
                subjects.append({
                    "subject_code": f"SUBJ{subject_id:03d}",
                    "branch": branch,
                    "sem": sem,
                    "subject_type": subject_type,
                    "credits": int(credits),
                    "faculty_code": faculty_code
                })
                subject_id += 1
    data = {"subjects": subjects}
    
    filepath = os.path.join(RESOURCE_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Subjects JSON saved to {filepath}")

if __name__ == "__main__":
    create_rooms_json()
    faculty_ids = create_faculty_json()  # Get faculty IDs
    create_subjects_json(faculty_ids)