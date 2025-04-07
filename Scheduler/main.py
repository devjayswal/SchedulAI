import os
from dotenv import load_dotenv
from fastapi import FastAPI
from routes.user_routes import router as user_router
from routes.faculty_routes import router as faculty_router
from routes.classroom_routes import router as classroom_router
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from utils.log_generator import log_generator
import time

# Load environment variables
load_dotenv()

PORT = int(os.getenv("PORT", 8000))  # Default to 8000 if not found

app = FastAPI(title="Timetable Scheduler API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(user_router)
app.include_router(faculty_router)
app.include_router(classroom_router)

@app.get("/")
async def root():
    return {"message": "Welcome to Timetable Scheduler API"}

@app.get("/logs")
def stream_logs():
    return StreamingResponse(log_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
