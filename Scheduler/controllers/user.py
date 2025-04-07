from bson import ObjectId
from Scheduler.utils.database import db
from models.user import UserCreate, UserResponse

user_collection = db["users"]

async def create_user(user: UserCreate) -> UserResponse:
    new_user = await user_collection.insert_one(user.dict())
    return UserResponse(id=str(new_user.inserted_id), **user.dict())

async def get_all_users():
    users = await user_collection.find({}, {"_id": 1}).to_list(100)
    return {"user_ids": [str(u["_id"]) for u in users]}

async def get_user(user_id: str) -> UserResponse:
    user = await user_collection.find_one({"_id": ObjectId(user_id)})
    if not user:
        return None
    return UserResponse(id=str(user["_id"]), **user)

async def update_user(user_id: str, user: UserCreate):
    result = await user_collection.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": user.dict()}
    )
    return result.modified_count > 0

async def delete_user(user_id: str):
    result = await user_collection.delete_one({"_id": ObjectId(user_id)})
    return result.deleted_count > 0
