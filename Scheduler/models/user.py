class User:
    def __init__(self, id=None, name="", role="", institute="", institute_mail=""):
        self.id = id  # MongoDB _id
        self.name = name
        self.role = role  # Example: "Admin", "Faculty", "Student"
        self.institute = institute
        self.institute_mail = institute_mail

    def __repr__(self):
        return f"User({self.name}, {self.role}, {self.institute})"
