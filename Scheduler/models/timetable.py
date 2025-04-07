from pydantic import BaseModel, Field
from typing import List, Dict, Any
import datetime
from .course import Course as course
from .faculty import Faculty as faculty
from .classroom import Classroom as classroom
from .user import User as user

class TimetableEntry(BaseModel):
    row: int
    col: int
    data: Dict[str, Any]  # Stores actual timetable entry

class Timetable(BaseModel):
    id: str | None = Field(default=None, alias="_id")
    branchs: str
    sems: int
    timetables: List[TimetableEntry]  # Stores multiple entries
    faculty:List[faculty]
    courses: List[course]
    classrooms: List[classroom]
    user:user
    class Config:
        from_attributes = True   
    


 
