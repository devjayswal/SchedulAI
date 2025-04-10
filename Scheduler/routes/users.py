from fastapi import APIRouter, HTTPException
from controllers.user import (
    create_user,
    get_all_users,
    get_user,
    update_user,
    delete_user,
)
from models.user import User  # SQLAlchemy model

router = APIRouter(prefix="/user", tags=["User"])

@router.post("/")
async def create_user_route(user: dict):  # Accepts a dictionary instead of a Pydantic model
    return await create_user(user)

@router.get("/")
async def get_all_users_route():
    return await get_all_users()

@router.get("/{user_id}")
async def get_user_route(user_id: str):
    user = await get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.put("/{user_id}")
async def update_user_route(user_id: str, user: dict):  # Accepts a dictionary
    if not await update_user(user_id, user):
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "User updated successfully"}

@router.delete("/{user_id}")
async def delete_user_route(user_id: str):
    if not await delete_user(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "User deleted successfully"}
