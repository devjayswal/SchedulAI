from fastapi import APIRouter, HTTPException
from controllers.classroom_controller import (
    create_classroom,
    get_all_classrooms,
    get_classroom,
    update_classroom,
    delete_classroom,
)
from models.classroom import ClassroomCreate, ClassroomResponse

router = APIRouter(prefix="/classroom", tags=["Classroom"])

@router.post("/", response_model=ClassroomResponse)
async def create_classroom_route(classroom: ClassroomCreate):
    return await create_classroom(classroom)

@router.get("/")
async def get_all_classrooms_route():
    return await get_all_classrooms()

@router.get("/{classroom_id}", response_model=ClassroomResponse)
async def get_classroom_route(classroom_id: str):
    classroom = await get_classroom(classroom_id)
    if not classroom:
        raise HTTPException(status_code=404, detail="Classroom not found")
    return classroom

@router.put("/{classroom_id}")
async def update_classroom_route(classroom_id: str, classroom: ClassroomCreate):
    if not await update_classroom(classroom_id, classroom):
        raise HTTPException(status_code=404, detail="Classroom not found")
    return {"message": "Classroom updated successfully"}

@router.delete("/{classroom_id}")
async def delete_classroom_route(classroom_id: str):
    if not await delete_classroom(classroom_id):
        raise HTTPException(status_code=404, detail="Classroom not found")
    return {"message": "Classroom deleted successfully"}
