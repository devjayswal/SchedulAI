class Classroom:
    def __init__(self, id=None, code="", type=""):
        self.id = id  # MongoDB _id
        self.code = code
        self.type = type  # Either "Theory" or "Computer Lab"

    def __repr__(self):
        return f"Classroom({self.code}, Type: {self.type})"
