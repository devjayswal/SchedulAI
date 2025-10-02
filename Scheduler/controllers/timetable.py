from utils.database import db
from bson import ObjectId
import asyncio
from utils.job_manager import create_job, get_queue, set_status, get_status
from ppo.cnn_train import run_cnn_training
from ppo.training import run_training
from ppo.continuous_training import run_continuous_training
from ppo.config import get_config
from serializers.jsonToClass import jsonToClass
from utils.config_processor import config_processor

timetable_collection = db["Timetables"]


import os
import traceback

async def _run_job(job_id: str, queue: asyncio.Queue, data, use_enhanced: bool = True, use_continuous: bool = True):
    """Runs the training job and logs output to a file."""
    
    log_dir = f"logs/{job_id}"
    os.makedirs(log_dir, exist_ok=True)  # Create directory if not exists
    log_file = os.path.join(log_dir, "log.log")

    try:
        set_status(job_id, "running")

        with open(log_file, "a") as log:
            log.write(f"Job {job_id} started...\n")
            await queue.put(f"Job {job_id} started...")

            if use_continuous:
                # Use continuous training system (default and optimized)
                log.write("Using Continuous Training System (Default & Optimized)...\n")
                await queue.put("🔄 Using Continuous Training System (Default & Optimized)...")
                
                # Get enhanced configuration
                enhanced_config = get_config("training")
                
                # Extract training parameters from data if available
                training_config = getattr(data, 'training_config', {})
                use_cnn = training_config.get('use_enhanced_cnn', None)  # Auto-detect if None
                n_envs = training_config.get('n_envs', enhanced_config['n_envs'])
                total_timesteps = training_config.get('total_timesteps', enhanced_config['total_timesteps'])
                
                log.write(f"Continuous Training Config: CNN={use_cnn} (auto-detected), N_ENVS={n_envs}, TIMESTEPS={total_timesteps}\n")
                await queue.put(f"Continuous Config: CNN={use_cnn} (auto-detected), Parallel Envs={n_envs}, Timesteps={total_timesteps}")
                
                async for msg in run_continuous_training(data, job_id, use_cnn, n_envs, total_timesteps):
                    log.write(msg + "\n")  # Write log to file
                    await queue.put(msg)   # Stream log to queue
                    
            elif use_enhanced:
                # Use enhanced training system (one-time training)
                log.write("Using Enhanced Training System (One-time)...\n")
                await queue.put("[ENHANCED] Using Enhanced Training System (One-time)...")
                
                # Get enhanced configuration
                enhanced_config = get_config("training")
                
                # Extract training parameters from data if available
                training_config = getattr(data, 'training_config', {})
                use_cnn = training_config.get('use_enhanced_cnn', True)
                n_envs = training_config.get('n_envs', enhanced_config['n_envs'])
                total_timesteps = training_config.get('total_timesteps', enhanced_config['total_timesteps'])
                
                log.write(f"Enhanced Training Config: CNN={use_cnn}, N_ENVS={n_envs}, TIMESTEPS={total_timesteps}\n")
                await queue.put(f"Enhanced Config: CNN={use_cnn}, Parallel Envs={n_envs}, Timesteps={total_timesteps}")
                
                async for msg in run_training(data, job_id, use_cnn, n_envs, total_timesteps):
                    log.write(msg + "\n")  # Write log to file
                    await queue.put(msg)   # Stream log to queue
            else:
                # Use original CNN training (legacy)
                log.write("Using Legacy CNN Training...\n")
                await queue.put("Using Legacy CNN Training...")
                
                async for msg in run_cnn_training(data, job_id):
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
async def create_timetable(timetable_data):  # Accepts Pydantic model or dict
    # Convert Pydantic model to dict if needed
    if hasattr(timetable_data, 'dict'):
        timetable_data = timetable_data.dict()
    elif hasattr(timetable_data, 'model_dump'):
        timetable_data = timetable_data.model_dump()
    
    # Ensure all config sections are dictionaries, not None
    if timetable_data.get("schedule_config") is None:
        timetable_data["schedule_config"] = {}
    if timetable_data.get("infrastructure_config") is None:
        timetable_data["infrastructure_config"] = {}
    if timetable_data.get("training_config") is None:
        timetable_data["training_config"] = {}
    if timetable_data.get("classrooms") is None:
        timetable_data["classrooms"] = []
    
    # Process dynamic configuration
    processed_config = config_processor.process_request(timetable_data)
    
    # Extract environment and training configs
    env_config = config_processor.get_environment_config(processed_config)
    training_config = config_processor.get_training_config(processed_config)
    
    # Add processed configs to the data
    timetable_data["processed_config"] = processed_config
    timetable_data["env_config"] = env_config
    timetable_data["training_config"] = training_config
    
    # Continuous training is now the default (always True)
    use_continuous = True  # Always use continuous training
    use_enhanced = training_config.get('use_enhanced_training', True)
    
    # Auto-detect fastest training method
    from ppo.continuous_training import ContinuousTrainingManager
    continuous_manager = ContinuousTrainingManager()
    use_cnn, _ = continuous_manager.get_fastest_training_method()
    
    # Update training config with auto-detected method
    training_config['use_enhanced_cnn'] = use_cnn
    training_config['use_continuous_training'] = use_continuous
    training_config['use_enhanced_training'] = use_enhanced
    
    # Serialize the data to class if needed
    timetable_object = jsonToClass(timetable_data)

    # Create job with metadata
    job_id = create_job("timetable", timetable_object)  # ✅ Now correct
    queue = get_queue(job_id)

    # Start training with continuous system (default and optimized)
    asyncio.create_task(_run_job(job_id, queue, timetable_object, use_enhanced, use_continuous))  # ✅ Continuous training
    return {
        "job_id": job_id, 
        "continuous_training": use_continuous, 
        "enhanced_training": use_enhanced,
        "auto_detected_cnn": use_cnn,
        "training_method": "continuous_optimized"
    }



# Get Generated Timetable by Job ID
async def get_generated_timetable_by_job_id(job_id: str):
    """Get the generated timetable by job ID."""
    generated_collection = db["GeneratedTimetables"]
    timetable = await generated_collection.find_one({"job_id": job_id})
    if timetable:
        timetable["_id"] = str(timetable["_id"])  # Convert ObjectId to string
    return timetable

# Get Specific Timetable
async def get_timetable_by_id(timetable_id: str):
    timetable = await timetable_collection.find_one({"_id": ObjectId(timetable_id)})
    if not timetable:
        return None
    return {"id": str(timetable["_id"]), "name": timetable["name"], "description": timetable.get("description")}

# Get Full Timetable Data
async def get_timetable_data(timetable_id: str):
    """Get complete timetable data including all timetables (student, faculty, classroom)"""
    timetable = await timetable_collection.find_one({"_id": ObjectId(timetable_id)})
    if not timetable:
        return None
    
    # Return the complete timetable data
    return {
        "id": str(timetable["_id"]),
        "name": timetable.get("name", "Unnamed Timetable"),
        "description": timetable.get("description", ""),
        "data": timetable.get("data", {}),
        "created_at": timetable.get("created_at"),
        "status": timetable.get("status", "unknown")
    }

# Export Timetable to CSV
async def export_timetable_csv(timetable_id: str, timetable_type: str = "master"):
    """Export timetable data to CSV format"""
    timetable = await timetable_collection.find_one({"_id": ObjectId(timetable_id)})
    if not timetable:
        return None
    
    # This would generate CSV content based on the timetable data
    # For now, return a placeholder response
    return {
        "message": f"CSV export for {timetable_type} timetable",
        "filename": f"{timetable_type}_timetable.csv",
        "data": "CSV content would be generated here"
    }

# Regenerate Timetable
async def regenerate_timetable(timetable_id: str):
    """Regenerate a timetable using the same input data but with fresh AI training"""
    timetable = await timetable_collection.find_one({"_id": ObjectId(timetable_id)})
    if not timetable:
        return None
    
    # Get the original input data
    original_data = timetable.get("data", {})
    
    # Create a new job for regeneration
    job_id = create_job("timetable_regenerate", original_data)
    queue = get_queue(job_id)
    
    # Start the regeneration process with enhanced training
    asyncio.create_task(_run_job(job_id, queue, original_data, use_enhanced=True))
    
    return {
        "message": "Timetable regeneration started",
        "job_id": job_id,
        "status": "regenerating"
    }

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
