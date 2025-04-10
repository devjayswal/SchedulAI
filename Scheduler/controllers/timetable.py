from models.Timetable import Timetable
from utils.database import db
from bson import ObjectId
import asyncio
from utils.job_manager import create_job, get_queue, set_status, get_status
from ppo.train import run_training
from serializers.jsonToClass import jsonToClass

timetable_collection = db["timetables"]


async def _run_job(job_id: str, queue: asyncio.Queue, data):
        try:
            set_status(job_id, "running")
            # run_training yields log messages
            async for msg in run_training(data):
                await queue.put(msg)
            set_status(job_id, "completed")
            await queue.put("DONE")
        except Exception as e:
            set_status(job_id, "failed")
            await queue.put(f"ERROR: {e}")
            await queue.put("DONE")

# Create Timetable (Async Status Update)
# Create Timetable (Async Status Update)
async def create_timetable(timetable_data: dict):  # Accepts raw dictionary
    # Serialize the data to class if needed
    timetable_object = jsonToClass(timetable_data)

    job_id = await create_job("timetable", timetable_data)  # No need to call .dict()
    queue = get_queue(job_id)

    asyncio.create_task(_run_job(job_id, queue, timetable_data))  # Fix missing job_id
    return {"job_id": job_id}


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
async def update_timetable(timetable_id: str, updated_data: dict):
    result = await timetable_collection.update_one({"_id": ObjectId(timetable_id)}, {"$set": updated_data})
    return result.modified_count > 0


# Delete Timetable
async def delete_timetable(timetable_id: str):
    result = await timetable_collection.delete_one({"_id": ObjectId(timetable_id)})
    return result.deleted_count > 0
