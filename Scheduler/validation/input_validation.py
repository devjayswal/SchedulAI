# models/schedule_input.py
from pydantic import BaseModel, EmailStr, Field
from typing import List, Literal, Optional, Dict, Any

class CourseIn(BaseModel):
    subject_code: str
    subject_name: str
    subject_type: Literal["theory", "lab"]
    credits: int
    faculty_id: str

class BranchIn(BaseModel):
    branch_name: str
    semester: int
    courses: List[CourseIn]

class FacultyIn(BaseModel):
    id: str
    name: str
    email: EmailStr

class ClassroomIn(BaseModel):
    id: str
    name: Optional[str] = None
    type: Literal["theory", "lab"]
    capacity: Optional[int] = None

class ScheduleConfig(BaseModel):
    weekdays: List[str] = Field(default=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
    time_slots: List[str] = Field(default=[
        "09:00-10:00", "10:00-11:00", "11:00-12:00", 
        "12:00-13:00", "14:00-15:00", "15:00-16:00"
    ])
    lunch_break: str = Field(default="12:00-13:00")
    max_daily_slots: int = Field(default=6)
    max_weekly_slots: int = Field(default=30)

class InfrastructureConfig(BaseModel):
    default_classroom_count: int = Field(default=5)
    default_lab_count: int = Field(default=2)
    default_theory_room_count: int = Field(default=3)
    classroom_capacity: Dict[str, int] = Field(default={"theory": 50, "lab": 30})

class TrainingConfig(BaseModel):
    learning_rate: float = Field(default=3e-4)
    batch_size: int = Field(default=64)
    n_steps: int = Field(default=1024)
    n_epochs: int = Field(default=4)
    total_timesteps: int = Field(default=500000)
    use_enhanced_rewards: bool = Field(default=True)
    use_curriculum_learning: bool = Field(default=True)

class ScheduleInput(BaseModel):
    # Dynamic configuration sections
    schedule_config: Optional[ScheduleConfig] = None
    infrastructure_config: Optional[InfrastructureConfig] = None
    training_config: Optional[TrainingConfig] = None
    
    # Core data
    branches: List[BranchIn]
    faculty: List[FacultyIn]
    classrooms: List[ClassroomIn]
    
    # Legacy fields for backward compatibility
    weekdays: Optional[List[str]] = None
    time_slots: Optional[List[str]] = None
