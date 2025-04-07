from pydantic import BaseModel, Field
from bson import ObjectId

class Faculty(BaseModel):
    id: str | None = Field(default=None, alias="_id")
    full_name: str
    short_name: str
    courses: list[str]  # List of course IDs
    
    class Config:
        from_attributes = True
