from models.timetable import TimetableCreate, TimetableResponse, TimetableStatus
from Scheduler.utils.database import db
from bson import ObjectId

timetable_collection = db["timetables"]

# Create Timetable (Async Status Update)
async def create_timetable(timetable_data: TimetableCreate):
    result = await timetable_collection.insert_one(timetable_data.dict())
    timetable_id = str(result.inserted_id)
    return {"status": "processing", "message": "Timetable generation in progress", "timetable_id": timetable_id}

# Get Specific Timetable
async def get_timetable_by_id(timetable_id: str):
    timetable = await timetable_collection.find_one({"_id": ObjectId(timetable_id)})
    if not timetable:
        return None
    return {"id": str(timetable["_id"]), "name": timetable["name"], "description": timetable.get("description")}

# Get All Timetable IDs
async def get_all_timetables():
    timetables = await timetable_collection.find({}, {"_id": 1}).to_list(100)
    return {"timetable_ids": [str(t["_id"]) for t in timetables]}

# Update Timetable
async def update_timetable(timetable_id: str, updated_data: TimetableCreate):
    result = await timetable_collection.update_one({"_id": ObjectId(timetable_id)}, {"$set": updated_data.dict()})
    return result.modified_count > 0

# Delete Timetable
async def delete_timetable(timetable_id: str):
    result = await timetable_collection.delete_one({"_id": ObjectId(timetable_id)})
    return result.deleted_count > 0
