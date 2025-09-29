from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from utils.job_manager import (
    get_job_summary, get_all_jobs, get_status, get_progress, 
    get_logs, get_metadata, add_log, update_progress
)
import json
import asyncio
from typing import Optional

router = APIRouter(prefix="/status", tags=["Status"])

@router.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Get complete status information for a specific job."""
    try:
        summary = get_job_summary(job_id)
        if "error" in summary:
            raise HTTPException(status_code=404, detail=summary["error"])
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/job/{job_id}/progress")
async def get_job_progress(job_id: str):
    """Get progress information for a specific job."""
    try:
        progress = get_progress(job_id)
        if not progress:
            raise HTTPException(status_code=404, detail="Job not found")
        return progress
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/job/{job_id}/logs")
async def get_job_logs(job_id: str, limit: int = 50):
    """Get recent log entries for a specific job."""
    try:
        logs = get_logs(job_id, limit)
        if logs is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return {"job_id": job_id, "logs": logs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/job/{job_id}/stream")
async def stream_job_logs(job_id: str):
    """Stream real-time log updates for a job."""
    try:
        from utils.job_manager import get_queue
        
        async def generate():
            queue = get_queue(job_id)
            while True:
                try:
                    # Wait for new log entry with timeout
                    message = await asyncio.wait_for(queue.get(), timeout=30.0)
                    if message == "DONE":
                        yield f"data: {json.dumps({'type': 'complete', 'message': 'Job completed'})}\n\n"
                        break
                    else:
                        yield f"data: {json.dumps({'type': 'log', 'message': message})}\n\n"
                except asyncio.TimeoutError:
                    # Send heartbeat to keep connection alive
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                    break
        
        return StreamingResponse(
            generate(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/jobs")
async def get_all_job_status():
    """Get status summary of all jobs."""
    try:
        jobs = get_all_jobs()
        return {"jobs": jobs, "total": len(jobs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/jobs/active")
async def get_active_jobs():
    """Get only active (pending/running) jobs."""
    try:
        all_jobs = get_all_jobs()
        active_jobs = [job for job in all_jobs if job["status"] in ["pending", "running"]]
        return {"jobs": active_jobs, "total": len(active_jobs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/jobs/completed")
async def get_completed_jobs():
    """Get only completed jobs."""
    try:
        all_jobs = get_all_jobs()
        completed_jobs = [job for job in all_jobs if job["status"] in ["completed", "failed"]]
        return {"jobs": completed_jobs, "total": len(completed_jobs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/job/{job_id}")
async def cleanup_job(job_id: str):
    """Clean up a completed job."""
    try:
        from utils.job_manager import cleanup_job
        cleanup_job(job_id)
        return {"message": "Job cleaned up successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "status-monitor"}
