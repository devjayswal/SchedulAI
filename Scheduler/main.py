import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
# from routes.users import router as user_router
# from routes.faculty import router as faculty_router
# from routes.classroom import router as classroom_router
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from utils.log_generator import log_generator
import time
from utils.job_manager import create_job, get_queue, set_status, get_status
from routes.timetable import router as timetable_router
from routes.status import router as status_router
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Load environment variables
load_dotenv()

PORT = int(os.getenv("PORT", 8000))  # Default to 8000 if not found

app = FastAPI(title="Timetable Scheduler API", version="1.0.0")
app.mount("/static", StaticFiles(directory="public"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# app.include_router(user_router)
# app.include_router(faculty_router)
# app.include_router(classroom_router)
app.include_router(timetable_router)
app.include_router(status_router)

@app.get("/")
async def root():
    return FileResponse(os.path.join("public", "index.html"))

@app.get("/logs/{job_id}")
async def stream_logs(job_id: str):
    try:
        queue = get_queue(job_id)
    except KeyError:
        raise HTTPException(404, "Job not found")

    async def event_generator():
        # First send status
        yield f"data: STATUS:{get_status(job_id)}\n\n"
        # Then stream log lines until "DONE"
        while True:
            msg = await queue.get()
            yield f"data: {msg}\n\n"
            if msg == "DONE":
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
