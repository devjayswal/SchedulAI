from pydantic import BaseModel, Field

class Course(BaseModel):
    id: str | None = Field(default=None, alias="_id")
    subject_code: str
    branch: str
    sem: int
    subject_name: str
    subject_type: str  # "theory" or "practical"
    credits: int
    faculty_code: str  # Can be empty if not assigned

    class Config:
        from_attributes = True
