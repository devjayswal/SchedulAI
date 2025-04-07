from fastapi import APIRouter, HTTPException
from Scheduler.utils.database import db
from models.course import CourseCreate, CourseResponse
from bson import ObjectId

router = APIRouter(prefix="/courses", tags=["Courses"])

@router.post("/", response_model=CourseResponse)
async def create_course(course: CourseCreate):
    """Create a new course."""
    new_course = await db["courses"].insert_one(course.dict())
    return {"id": str(new_course.inserted_id), **course.dict()}

@router.get("/")
async def get_all_courses():
    """Retrieve all course IDs."""
    courses = await db["courses"].find({}, {"_id": 1}).to_list(100)
    return {"course_ids": [str(c["_id"]) for c in courses]}

@router.get("/{course_id}", response_model=CourseResponse)
async def get_course(course_id: str):
    """Retrieve a course by ID."""
    course = await db["courses"].find_one({"_id": ObjectId(course_id)})
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    return {"id": str(course["_id"]), **course}

@router.put("/{course_id}")
async def update_course(course_id: str, course: CourseCreate):
    """Update a course."""
    result = await db["courses"].update_one({"_id": ObjectId(course_id)}, {"$set": course.dict()})
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Course not found")
    return {"message": "Course updated successfully"}

@router.delete("/{course_id}")
async def delete_course(course_id: str):
    """Delete a course."""
    result = await db["courses"].delete_one({"_id": ObjectId(course_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Course not found")
    return {"message": "Course deleted successfully"}
