from utils.database import db
from bson import ObjectId
import asyncio
from utils.job_manager import create_job, get_queue, set_status, get_status
from ppo.train import run_training
from serializers.jsonToClass import jsonToClass

timetable_collection = db["Timetables"]


import os
import traceback

async def _run_job(job_id: str, queue: asyncio.Queue, data):
    """Runs the training job and logs output to a file."""
    
    log_dir = f"logs/{job_id}"
    os.makedirs(log_dir, exist_ok=True)  # Create directory if not exists
    log_file = os.path.join(log_dir, "log.log")

    try:
        set_status(job_id, "running")

        with open(log_file, "a") as log:
            log.write(f"Job {job_id} started...\n")
            await queue.put(f"Job {job_id} started...")

            async for msg in run_training(data,job_id):
                log.write(msg + "\n")  # Write log to file
                await queue.put(msg)   # Stream log to queue

            set_status(job_id, "completed")
            log.write("Training completed.\n")
            await queue.put("DONE")

    except Exception as e:
        set_status(job_id, "failed")
        error_msg = f"ERROR: {e}\n{traceback.format_exc()}"
        
        with open(log_file, "a") as log:
            log.write(error_msg + "\n")
        
        await queue.put(error_msg)
        await queue.put("DONE")


# Create Timetable (Async Status Update)
async def create_timetable(timetable_data: dict):  # Accepts raw dictionary
    # Serialize the data to class if needed
    timetable_object = jsonToClass(timetable_data)

    # Create job with metadata
    job_id = create_job("timetable", timetable_object)  # ✅ Now correct
    queue = get_queue(job_id)

    asyncio.create_task(_run_job(job_id, queue, timetable_object))  # ✅ Proper async execution
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
