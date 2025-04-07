from pydantic import BaseModel, EmailStr, Field

class User(BaseModel):
    id: str | None = Field(default=None, alias="_id")
    name: str
    role: str  # Example: "Admin", "Faculty", "Student"
    institute: str
    institute_mail: EmailStr

    class Config:
        from_attributes = True
