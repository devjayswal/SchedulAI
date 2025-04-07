from fastapi import APIRouter, HTTPException
from controllers.user_controller import (
    create_user,
    get_all_users,
    get_user,
    update_user,
    delete_user,
)
from models.user import UserCreate, UserResponse

router = APIRouter(prefix="/user", tags=["User"])

@router.post("/", response_model=UserResponse)
async def create_user_route(user: UserCreate):
    return await create_user(user)

@router.get("/")
async def get_all_users_route():
    return await get_all_users()

@router.get("/{user_id}", response_model=UserResponse)
async def get_user_route(user_id: str):
    user = await get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.put("/{user_id}")
async def update_user_route(user_id: str, user: UserCreate):
    if not await update_user(user_id, user):
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "User updated successfully"}

@router.delete("/{user_id}")
async def delete_user_route(user_id: str):
    if not await delete_user(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "User deleted successfully"}
