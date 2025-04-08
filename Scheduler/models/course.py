class Course:
    def __init__(self, id=None, subject_code="", branch="", sem=0, subject_name="", subject_type="", credits=0, faculty_code=""):
        self.id = id  # MongoDB _id
        self.subject_code = subject_code
        self.branch = branch
        self.sem = sem
        self.subject_name = subject_name
        self.subject_type = subject_type  # "theory" or "practical"
        self.credits = credits
        self.faculty_code = faculty_code  # Can be empty if not assigned

    def __repr__(self):
        return f"Course({self.subject_name}, {self.subject_type}, Credits: {self.credits})"
