from bson import ObjectId
from utils.database import db
from models.user import User  # Using your existing User class

user_collection = db["users"]

async def create_user(user: User):
    new_user = await user_collection.insert_one(user.__dict__)  # Direct insertion
    return {"id": str(new_user.inserted_id), **user.__dict__}  # Return with MongoDB ID

async def get_all_users():
    users = await user_collection.find({}, {"_id": 1}).to_list(100)
    return {"user_ids": [str(u["_id"]) for u in users]}  # Return user IDs only

async def get_user(user_id: str):
    user = await user_collection.find_one({"_id": ObjectId(user_id)})
    if not user:
        return None
    user["id"] = str(user["_id"])  # Convert MongoDB _id to string
    del user["_id"]  # Remove raw ObjectId from response
    return user

async def update_user(user_id: str, user: User):
    result = await user_collection.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": user.__dict__}
    )
    return result.modified_count > 0  # Return success status

async def delete_user(user_id: str):
    result = await user_collection.delete_one({"_id": ObjectId(user_id)})
    return result.deleted_count > 0  # Return success status
