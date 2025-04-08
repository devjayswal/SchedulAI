# models/schedule_input.py
from pydantic import BaseModel, EmailStr
from typing import List, Literal

class CourseIn(BaseModel):
    subject_code: str
    subject_name: str
    subject_type: Literal["theory", "lab"]
    credits: int
    faculty_id: str

class BranchIn(BaseModel):
    branch_name: str
    semester: int
    courses: List[CourseIn]

class FacultyIn(BaseModel):
    id: str
    name: str
    email: EmailStr

class ClassroomIn(BaseModel):
    id: str
    type: Literal["theory", "lab"]

class ScheduleInput(BaseModel):
    weekdays: List[str]
    time_slots: List[str]
    branches: List[BranchIn]
    faculty: List[FacultyIn]
    classrooms: List[ClassroomIn]
