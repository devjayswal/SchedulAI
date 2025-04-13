# models/Branch.py
from typing import List
from models.Course import Course

class Branch:
    def __init__(self, branch_name: str, semester: int, courses: List[Course]):
        self.branch_name = branch_name
        self.semester = semester
        self.courses = courses
