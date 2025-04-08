class Faculty:
    def __init__(self, id=None, full_name="", short_name="", courses=None):
        self.id = id  # MongoDB _id
        self.full_name = full_name
        self.short_name = short_name
        self.courses = courses if courses is not None else []  # List of course IDs

    def __repr__(self):
        return f"Faculty({self.full_name}, {self.short_name}, Courses: {len(self.courses)})"
