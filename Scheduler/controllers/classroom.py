from bson import ObjectId
from Scheduler.utils.database import db
from models.Classroom import ClassroomCreate, ClassroomResponse

classroom_collection = db["classrooms"]

async def create_classroom(classroom: ClassroomCreate) -> ClassroomResponse:
    new_classroom = await classroom_collection.insert_one(classroom.dict())
    return ClassroomResponse(id=str(new_classroom.inserted_id), **classroom.dict())

async def get_all_classrooms():
    classrooms = await classroom_collection.find({}, {"_id": 1}).to_list(100)
    return {"classroom_ids": [str(c["_id"]) for c in classrooms]}

async def get_classroom(classroom_id: str) -> ClassroomResponse:
    classroom = await classroom_collection.find_one({"_id": ObjectId(classroom_id)})
    if not classroom:
        return None
    return ClassroomResponse(id=str(classroom["_id"]), **classroom)

async def update_classroom(classroom_id: str, classroom: ClassroomCreate):
    result = await classroom_collection.update_one(
        {"_id": ObjectId(classroom_id)},
        {"$set": classroom.dict()}
    )
    return result.modified_count > 0

async def delete_classroom(classroom_id: str):
    result = await classroom_collection.delete_one({"_id": ObjectId(classroom_id)})
    return result.deleted_count > 0
