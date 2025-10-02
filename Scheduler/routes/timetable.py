from fastapi import APIRouter, HTTPException, Body
from controllers.timetable import create_timetable,get_timetable_by_id,get_all_timetables,update_timetable,delete_timetable,get_timetable_data,export_timetable_csv,regenerate_timetable,get_generated_timetable_by_job_id

from models.Timetable import Timetable
from validation.input_validation import ScheduleInput


router = APIRouter(prefix="/timetable", tags=["Timetable"])

@router.post("/", response_model=dict)
async def create_timetable_route(payload:ScheduleInput=Body(...)):
    """Create a new timetable using continuous training (default and optimized)."""
    # Validate the input data
    result = await create_timetable(payload)
    return {
        "job_id": result["job_id"], 
        "status": "continuous training job created", 
        "continuous_training": result["continuous_training"],
        "enhanced_training": result["enhanced_training"],
        "auto_detected_cnn": result["auto_detected_cnn"],
        "training_method": result["training_method"]
    }

@router.post("/continuous", response_model=dict)
async def create_continuous_timetable_route(payload:ScheduleInput=Body(...)):
    """Create a new timetable using continuous training system."""
    # Force continuous training
    if hasattr(payload, 'training_config'):
        payload.training_config = payload.training_config or {}
        payload.training_config['use_continuous_training'] = True
        payload.training_config['use_enhanced_training'] = True
    else:
        payload.training_config = {'use_continuous_training': True, 'use_enhanced_training': True}
    
    result = await create_timetable(payload)
    return {"job_id": result["job_id"], "status": "continuous training job created", "continuous_training": True}

@router.post("/enhanced", response_model=dict)
async def create_enhanced_timetable_route(payload:ScheduleInput=Body(...)):
    """Create a new timetable using enhanced training system."""
    # Force enhanced training
    if hasattr(payload, 'training_config'):
        payload.training_config = payload.training_config or {}
        payload.training_config['use_enhanced_training'] = True
        payload.training_config['use_continuous_training'] = False
    else:
        payload.training_config = {'use_enhanced_training': True, 'use_continuous_training': False}
    
    result = await create_timetable(payload)
    return {"job_id": result["job_id"], "status": "enhanced training job created", "enhanced_training": True}

@router.post("/legacy", response_model=dict)
async def create_legacy_timetable_route(payload:ScheduleInput=Body(...)):
    """Create a new timetable using legacy training system."""
    # Force legacy training
    if hasattr(payload, 'training_config'):
        payload.training_config = payload.training_config or {}
        payload.training_config['use_enhanced_training'] = False
        payload.training_config['use_continuous_training'] = False
    else:
        payload.training_config = {'use_enhanced_training': False, 'use_continuous_training': False}
    
    result = await create_timetable(payload)
    return {"job_id": result["job_id"], "status": "legacy training job created", "enhanced_training": False}

@router.get("/{timetable_id}", response_model=None)
async def get_timetable_route(timetable_id: str):
    """Retrieve a specific timetable by ID."""
    timetable = await get_timetable_by_id(timetable_id)
    if not timetable:
        raise HTTPException(status_code=404, detail="Timetable not found")
    return timetable

@router.get("/{timetable_id}/data", response_model=None)
async def get_timetable_data_route(timetable_id: str):
    """Retrieve complete timetable data including all timetables."""
    timetable_data = await get_timetable_data(timetable_id)
    if not timetable_data:
        raise HTTPException(status_code=404, detail="Timetable not found")
    return timetable_data

@router.get("/{timetable_id}/export/{timetable_type}", response_model=None)
async def export_timetable_route(timetable_id: str, timetable_type: str = "master"):
    """Export timetable data to CSV format."""
    export_data = await export_timetable_csv(timetable_id, timetable_type)
    if not export_data:
        raise HTTPException(status_code=404, detail="Timetable not found")
    return export_data

@router.post("/{timetable_id}/regenerate", response_model=None)
async def regenerate_timetable_route(timetable_id: str):
    """Regenerate a timetable with the same input data."""
    result = await regenerate_timetable(timetable_id)
    if not result:
        raise HTTPException(status_code=404, detail="Timetable not found")
    return result

@router.get("/generated/{job_id}")
async def get_generated_timetable_route(job_id: str):
    """Get the generated timetable by job ID."""
    timetable = await get_generated_timetable_by_job_id(job_id)
    if not timetable:
        raise HTTPException(status_code=404, detail="Generated timetable not found")
    return timetable

@router.get("/")
async def get_all_timetables_route():
    """Retrieve all timetable IDs."""
    return await get_all_timetables()

@router.put("/{timetable_id}")
async def update_timetable_route(timetable_id: str, timetable: None):
    """Update an existing timetable."""
    updated = await update_timetable(timetable_id, timetable)
    if not updated:
        raise HTTPException(status_code=404, detail="Timetable not found")
    return {"message": "Timetable updated successfully"}

@router.delete("/{timetable_id}")
async def delete_timetable_route(timetable_id: str):
    """Delete a timetable."""
    deleted = await delete_timetable(timetable_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Timetable not found")
    return {"message": "Timetable deleted successfully"}
