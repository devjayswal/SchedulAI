from bson import ObjectId
from Scheduler.utils.database import db
from models.faculty import FacultyCreate, FacultyResponse

faculty_collection = db["faculties"]

async def create_faculty(faculty: FacultyCreate) -> FacultyResponse:
    new_faculty = await faculty_collection.insert_one(faculty.dict())
    return FacultyResponse(id=str(new_faculty.inserted_id), **faculty.dict())

async def get_all_faculties():
    faculties = await faculty_collection.find({}, {"_id": 1}).to_list(100)
    return {"faculty_ids": [str(f["_id"]) for f in faculties]}

async def get_faculty(faculty_id: str) -> FacultyResponse:
    faculty = await faculty_collection.find_one({"_id": ObjectId(faculty_id)})
    if not faculty:
        return None
    return FacultyResponse(id=str(faculty["_id"]), **faculty)

async def update_faculty(faculty_id: str, faculty: FacultyCreate):
    result = await faculty_collection.update_one(
        {"_id": ObjectId(faculty_id)},
        {"$set": faculty.dict()}
    )
    return result.modified_count > 0

async def delete_faculty(faculty_id: str):
    result = await faculty_collection.delete_one({"_id": ObjectId(faculty_id)})
    return result.deleted_count > 0
