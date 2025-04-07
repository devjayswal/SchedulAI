from fastapi import APIRouter, HTTPException
from controllers.timetable_controller import (
    create_timetable,
    get_timetable_by_id,
    get_all_timetables,
    update_timetable,
    delete_timetable,
)
from models.timetable import TimetableCreate, TimetableResponse, TimetableStatus

router = APIRouter(prefix="/timetable", tags=["Timetable"])

@router.post("/", response_model=TimetableStatus)
async def create_timetable_route(timetable: TimetableCreate):
    """Create a new timetable asynchronously (returns status updates)."""
    return await create_timetable(timetable)

@router.get("/{timetable_id}", response_model=TimetableResponse)
async def get_timetable_route(timetable_id: str):
    """Retrieve a specific timetable by ID."""
    timetable = await get_timetable_by_id(timetable_id)
    if not timetable:
        raise HTTPException(status_code=404, detail="Timetable not found")
    return timetable

@router.get("/")
async def get_all_timetables_route():
    """Retrieve all timetable IDs."""
    return await get_all_timetables()

@router.put("/{timetable_id}")
async def update_timetable_route(timetable_id: str, timetable: TimetableCreate):
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
