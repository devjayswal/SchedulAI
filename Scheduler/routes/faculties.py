from fastapi import APIRouter, HTTPException, Depends
from controllers.faculty_controller import (
    create_faculty,
    get_all_faculties,
    get_faculty,
    update_faculty,
    delete_faculty,
)
from models.faculty import FacultyCreate, FacultyResponse

router = APIRouter(prefix="/faculty", tags=["Faculty"])

@router.post("/", response_model=FacultyResponse)
async def create_faculty_route(faculty: FacultyCreate):
    return await create_faculty(faculty)

@router.get("/")
async def get_all_faculties_route():
    return await get_all_faculties()

@router.get("/{faculty_id}", response_model=FacultyResponse)
async def get_faculty_route(faculty_id: str):
    faculty = await get_faculty(faculty_id)
    if not faculty:
        raise HTTPException(status_code=404, detail="Faculty not found")
    return faculty

@router.put("/{faculty_id}")
async def update_faculty_route(faculty_id: str, faculty: FacultyCreate):
    if not await update_faculty(faculty_id, faculty):
        raise HTTPException(status_code=404, detail="Faculty not found")
    return {"message": "Faculty updated successfully"}

@router.delete("/{faculty_id}")
async def delete_faculty_route(faculty_id: str):
    if not await delete_faculty(faculty_id):
        raise HTTPException(status_code=404, detail="Faculty not found")
    return {"message": "Faculty deleted successfully"}
