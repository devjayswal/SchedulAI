from pydantic import BaseModel, Field

class Classroom(BaseModel):
    id: str | None = Field(default=None, alias="_id")
    code: str
    type: str  # Either "Theory" or "Computer Lab"
    
    class Config:
        from_attributes = True
