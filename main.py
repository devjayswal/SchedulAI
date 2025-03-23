import os
import json
import numpy as np
import gym
from gym import spaces
from collections import defaultdict

class TimetableEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, resource_dir="resources", output_dir="output"):
        super(TimetableEnv, self).__init__()
        
        # Load JSON files from resources directory
        self.resource_dir = resource_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.rooms = self._load_json("rooms.json")["rooms"]
        self.subjects = self._load_json("subjects.json")["subjects"]
        self.faculty = self._load_json("faculty.json")["faculty"]
        
        self.num_days = 5
        self.num_slots = 9
        self.num_rooms = len(self.rooms)

        # Create lookup dictionaries
        self.faculty_dict = {f["faculty_id"]: f["name"] for f in self.faculty}
        self.room_dict = {r["room_no"]: r for r in self.rooms}
        
        # Group subjects by branch and semester
        self.branch_sem_subjects = defaultdict(list)
        for subject in self.subjects:
            key = (subject["branch"], subject["sem"])
            self.branch_sem_subjects[key].append(subject)

        # Initialize base timetable grid
        self.timetable = np.empty((self.num_days, self.num_slots, self.num_rooms), dtype=object)
        self.reset_timetable()

        # Build required sessions for each subject
        self.remaining_sessions = self._initialize_sessions()

        # Create subject index mapping
        self.subject_list = [subj["subject_code"] for subj in self.subjects]
        self.num_subjects = len(self.subject_list)

        # Define action and observation spaces
        self.action_space = spaces.MultiDiscrete([
            self.num_subjects,  # subject index
            self.num_days,      # day index
            self.num_slots,     # slot index
            self.num_rooms      # room index
        ])

        timetable_size = self.num_days * self.num_slots * self.num_rooms
        rem_sessions_size = len(self.subject_list)
        total_size = timetable_size + rem_sessions_size
        self.observation_space = spaces.Box(
            low=0, 
            high=100, 
            shape=(total_size,), 
            dtype=np.float32
        )

    def _load_json(self, filename):
        filepath = os.path.join(self.resource_dir, filename)
        with open(filepath, "r") as f:
            return json.load(f)

    def _initialize_sessions(self):
        sessions = {}
        for subj in self.subjects:
            code = subj["subject_code"]
            credits = subj["credits"]
            subj_type = subj["subject_type"]
            
            if credits == 2:
                if subj_type == "theory":
                    sessions[code] = ["theory"] * 2
                elif subj_type == "practical":
                    sessions[code] = ["lab"]
                else:
                    sessions[code] = ["theory"] * 2
            elif credits == 3:
                # For 3-credit subjects, if mixed then 1 theory + 1 lab; otherwise 3 theory sessions.
                if subj_type == "mixed":
                    sessions[code] = ["theory", "lab"]
                else:
                    sessions[code] = ["theory"] * 3
            else:
                sessions[code] = []
        return sessions

    def generate_student_timetable(self):
        """Generate student timetable grouped by branch and semester"""
        timetable = {}
        
        for (branch, sem) in self.branch_sem_subjects.keys():
            key = f"{branch}_{sem}"
            timetable[key] = {}
            
            for day in range(self.num_days):
                day_schedule = []
                for slot in range(self.num_slots):
                    slot_info = []
                    
                    # Check all rooms for this time slot
                    for room in range(self.num_rooms):
                        assignment = self.timetable[day, slot, room]
                        if assignment is not None:
                            subject_code = assignment["subject_code"]
                            subject_info = next(
                                (s for s in self.subjects if s["subject_code"] == subject_code 
                                 and s["branch"] == branch and s["sem"] == sem),
                                None
                            )
                            if subject_info:
                                slot_info.append({
                                    "subject_code": subject_code,
                                    "faculty": self.faculty_dict[subject_info["faculty_code"]],
                                    "room_no": self.rooms[room]["room_no"],
                                    "is_lab": self.rooms[room]["is_lab"],
                                    "session_type": assignment["session_type"]
                                })
                    
                    day_schedule.append(slot_info)
                timetable[key][f"Day_{day+1}"] = day_schedule
                
        return timetable

    def generate_faculty_timetable(self):
        """Generate faculty-wise timetable"""
        faculty_schedule = {fac["faculty_id"]: {} for fac in self.faculty}
        
        for day in range(self.num_days):
            for slot in range(self.num_slots):
                for room in range(self.num_rooms):
                    assignment = self.timetable[day, slot, room]
                    if assignment is not None:
                        subject_code = assignment["subject_code"]
                        subject_info = next(s for s in self.subjects if s["subject_code"] == subject_code)
                        faculty_id = subject_info["faculty_code"]
                        
                        if f"Day_{day+1}" not in faculty_schedule[faculty_id]:
                            faculty_schedule[faculty_id][f"Day_{day+1}"] = [[] for _ in range(self.num_slots)]
                            
                        faculty_schedule[faculty_id][f"Day_{day+1}"][slot].append({
                            "subject_code": subject_code,
                            "branch": subject_info["branch"],
                            "sem": subject_info["sem"],
                            "room_no": self.rooms[room]["room_no"],
                            "session_type": assignment["session_type"]
                        })
                        
        return faculty_schedule

    def generate_room_timetable(self):
        """Generate room-wise timetable"""
        room_schedule = {room["room_no"]: {} for room in self.rooms}
        
        for day in range(self.num_days):
            for slot in range(self.num_slots):
                for room in range(self.num_rooms):
                    room_no = self.rooms[room]["room_no"]
                    if f"Day_{day+1}" not in room_schedule[room_no]:
                        room_schedule[room_no][f"Day_{day+1}"] = [[] for _ in range(self.num_slots)]
                        
                    assignment = self.timetable[day, slot, room]
                    if assignment is not None:
                        subject_code = assignment["subject_code"]
                        subject_info = next(s for s in self.subjects if s["subject_code"] == subject_code)
                        
                        room_schedule[room_no][f"Day_{day+1}"][slot].append({
                            "subject_code": subject_code,
                            "faculty": self.faculty_dict[subject_info["faculty_code"]],
                            "branch": subject_info["branch"],
                            "sem": subject_info["sem"],
                            "session_type": assignment["session_type"]
                        })
                    
        return room_schedule

    def save_timetables(self):
        """Save student, faculty, and room timetables as JSON files"""
        student_timetable = self.generate_student_timetable()
        faculty_timetable = self.generate_faculty_timetable()
        room_timetable = self.generate_room_timetable()
        
        with open(os.path.join(self.output_dir, "student_timetable.json"), "w") as f:
            json.dump(student_timetable, f, indent=4)
        
        with open(os.path.join(self.output_dir, "faculty_timetable.json"), "w") as f:
            json.dump(faculty_timetable, f, indent=4)
        
        with open(os.path.join(self.output_dir, "room_timetable.json"), "w") as f:
            json.dump(room_timetable, f, indent=4)
    
    def render(self, mode='human'):
        """Render all three timetable views"""
        self.save_timetables()
        print("Timetables saved successfully in the 'output' folder.")

    def reset_timetable(self):
        for d in range(self.num_days):
            for s in range(self.num_slots):
                for r in range(self.num_rooms):
                    self.timetable[d, s, r] = None

    def reset(self):
        self.reset_timetable()
        self.remaining_sessions = self._initialize_sessions()
        return self._get_obs()

    def _get_obs(self):
        timetable_obs = np.zeros((self.num_days, self.num_slots, self.num_rooms), dtype=np.int8)
        for d in range(self.num_days):
            for s in range(self.num_slots):
                for r in range(self.num_rooms):
                    if self.timetable[d, s, r] is not None:
                        timetable_obs[d, s, r] = 1
        timetable_obs = timetable_obs.flatten()
        rem_sessions_obs = np.array([len(self.remaining_sessions[code]) for code in self.subject_list])
        return np.concatenate([timetable_obs, rem_sessions_obs]).astype(np.float32)
    
    def is_valid_assignment(self, subject_code, day, slot, room_idx):
        # Prevent scheduling during lunch break (slots 1 and 2)
        if slot in [1, 2]:
            return False, "Slot is reserved for lunch break."

        # Ensure only one class is scheduled per time slot across all rooms
        for r in range(self.num_rooms):
            if self.timetable[day, slot, r] is not None:
                return False, "Only one class allowed per time slot."

        # Ensure the faculty is not teaching another class in the same slot
        faculty_id = next(s for s in self.subjects if s["subject_code"] == subject_code)["faculty_code"]
        for r in range(self.num_rooms):
            assignment = self.timetable[day, slot, r]
            if assignment:
                assigned_faculty = next(s for s in self.subjects if s["subject_code"] == assignment["subject_code"])["faculty_code"]
                if assigned_faculty == faculty_id:
                    return False, "Faculty conflict at this slot."
        
        # Check that the room is free at the given slot
        if self.timetable[day, slot, room_idx] is not None:
            return False, "Room is already occupied."

        return True, "Valid assignment."

    def step(self, action):
        subject_idx, day, slot, room_idx = action
        subject_code = self.subject_list[subject_idx]
        room = self.rooms[room_idx]
        room_is_lab = room["is_lab"]

        done = False
        reward = 0

        # Check if there are remaining sessions for this subject
        if len(self.remaining_sessions[subject_code]) == 0:
            reward -= 5
            return self._get_obs(), reward, done, {"info": "No remaining sessions for subject."}

        # Validate assignment constraints
        valid, message = self.is_valid_assignment(subject_code, day, slot, room_idx)
        if not valid:
            reward -= 5
            return self._get_obs(), reward, done, {"info": message}
        
        required_session = self.remaining_sessions[subject_code][0]

        # Check room suitability for lab sessions
        if required_session == "lab" and not room_is_lab:
            reward -= 5
            return self._get_obs(), reward, done, {"info": "Room is not a lab for lab session."}

        # All validations passed; assign the session
        assignment = {
            "subject_code": subject_code,
            "session_type": required_session,
            "room_no": room["room_no"]
        }
        self.timetable[day, slot, room_idx] = assignment
        self.remaining_sessions[subject_code].pop(0)
        reward += 10
        
        if required_session == "lab":
            if slot + 1 < self.num_slots:
                self.timetable[day, slot + 1, room_idx] = assignment
                reward += 5  # Extra reward for successful lab placement
            else:
                reward -= 5  # Penalize if lab doesn't fit properly

        # Bonus reward if subject sessions are fully scheduled for the week
        if len(self.remaining_sessions[subject_code]) == 0:
            reward += 5

        # If all sessions for all subjects are scheduled, finish the episode
        if all(len(sessions) == 0 for sessions in self.remaining_sessions.values()):
            done = True
            reward += 50

        return self._get_obs(), reward, done, {"info": "Assignment successful."}

def auto_assign(env):
    """
    Automatically assign all sessions using a greedy approach.
    It iterates over all subjects and attempts to assign each remaining session
    into the first valid time slot and room.
    """
    done = False
    progress = True

    # Loop until no more progress is possible or all sessions are scheduled.
    while not done and progress:
        progress = False
        # Try every subject
        for subject_idx, subject_code in enumerate(env.subject_list):
            # Continue trying to assign sessions for the subject
            while len(env.remaining_sessions[subject_code]) > 0:
                assigned = False
                # Try all possible day, slot, and room combinations
                for day in range(env.num_days):
                    for slot in range(env.num_slots):
                        for room_idx in range(env.num_rooms):
                            valid, _ = env.is_valid_assignment(subject_code, day, slot, room_idx)
                            # Check if this slot is suitable for the required session type
                            required_session = env.remaining_sessions[subject_code][0]
                            room = env.rooms[room_idx]
                            if valid and (required_session != "lab" or (required_session == "lab" and room["is_lab"])):
                                action = (subject_idx, day, slot, room_idx)
                                obs, reward, done, info = env.step(action)
                                print(f"Assigned {subject_code} ({required_session}) on Day {day+1}, Slot {slot+1}, Room {room['room_no']} | Reward: {reward}")
                                assigned = True
                                progress = True
                                break
                        if assigned:
                            break
                    if assigned:
                        break
                # If no valid slot was found for the current session, break out of the loop for this subject.
                if not assigned:
                    print(f"Could not assign remaining session for {subject_code}")
                    break
        # Check if all sessions are scheduled.
        if all(len(sessions) == 0 for sessions in env.remaining_sessions.values()):
            done = True

    print("\nAuto-assignment complete. Final Timetables:")
    env.render()

if __name__ == "__main__":
    env = TimetableEnv()
    obs = env.reset()
    print("Initial observation shape:", obs.shape)
    
    auto_assign(env)
