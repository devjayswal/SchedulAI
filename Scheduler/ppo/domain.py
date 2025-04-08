# ppo/domain.py
from dataclasses import dataclass
from typing import List

@dataclass
class Course:
    subject_code: str
    subject_name: str
    subject_type: str  # "theory" or "lab"
    credits: int
    faculty_id: str

@dataclass
class Branch:
    branch_name: str
    semester: int
    courses: List[Course]

@dataclass
class Faculty:
    id: str
    name: str
    email: str

@dataclass
class Classroom:
    id: str
    type: str  # "theory" or "lab"

@dataclass
class ScheduleConfig:
    weekdays: List[str]
    time_slots: List[str]
    branches: List[Branch]
    faculty: List[Faculty]
    classrooms: List[Classroom]
