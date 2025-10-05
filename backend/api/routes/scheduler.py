"""
API endpoints for task scheduler
"""

from fastapi import APIRouter, HTTPException
from typing import Optional, List
from pydantic import BaseModel, Field

from services.scheduler import DiscoveryScheduler

router = APIRouter(prefix="/scheduler", tags=["Scheduler"])

# Global scheduler instance
scheduler = DiscoveryScheduler()


class CronTaskRequest(BaseModel):
    """Request to create a cron-based task"""
    task_id: str = Field(..., description="Unique task identifier")
    name: str = Field(..., description="Human-readable task name")
    cron_expression: str = Field(..., description="Cron expression (e.g., '0 */6 * * *')")
    max_retries: int = Field(3, ge=0, le=10, description="Maximum retry attempts")


class IntervalTaskRequest(BaseModel):
    """Request to create an interval-based task"""
    task_id: str = Field(..., description="Unique task identifier")
    name: str = Field(..., description="Human-readable task name")
    hours: int = Field(0, ge=0, description="Interval in hours")
    minutes: int = Field(0, ge=0, description="Interval in minutes")
    seconds: int = Field(0, ge=0, description="Interval in seconds")
    max_retries: int = Field(3, ge=0, le=10, description="Maximum retry attempts")


@router.post("/start")
async def start_scheduler():
    """Start the task scheduler"""
    if scheduler.is_running:
        return {
            "status": "already_running",
            "message": "Scheduler is already running"
        }
    
    scheduler.start()
    
    return {
        "status": "started",
        "message": "Scheduler started successfully"
    }


@router.post("/stop")
async def stop_scheduler():
    """Stop the task scheduler"""
    if not scheduler.is_running:
        return {
            "status": "not_running",
            "message": "Scheduler is not running"
        }
    
    scheduler.stop()
    
    return {
        "status": "stopped",
        "message": "Scheduler stopped successfully"
    }


@router.get("/status")
async def get_scheduler_status():
    """Get scheduler status"""
    return {
        "is_running": scheduler.is_running,
        "total_tasks": len(scheduler.tasks),
        "tasks": scheduler.get_all_tasks()
    }


@router.get("/tasks")
async def get_all_tasks():
    """Get all scheduled tasks"""
    return {
        "tasks": scheduler.get_all_tasks()
    }


@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get status of a specific task"""
    status = scheduler.get_task_status(task_id)
    
    if not status:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found"
        )
    
    return status


@router.post("/tasks/cron")
async def create_cron_task(request: CronTaskRequest):
    """
    Create a new cron-based scheduled task
    
    Example cron expressions:
    - "0 */6 * * *" - Every 6 hours
    - "0 0 * * *" - Daily at midnight
    - "0 0 * * 0" - Weekly on Sunday
    - "*/30 * * * *" - Every 30 minutes
    """
    try:
        # For now, we'll create a dummy function
        # In production, this would be connected to actual discovery functions
        async def dummy_task():
            print(f"Executing task: {request.name}")
        
        scheduler.add_cron_task(
            task_id=request.task_id,
            name=request.name,
            func=dummy_task,
            cron_expression=request.cron_expression,
            max_retries=request.max_retries
        )
        
        return {
            "status": "created",
            "task": scheduler.get_task_status(request.task_id)
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/tasks/interval")
async def create_interval_task(request: IntervalTaskRequest):
    """Create a new interval-based scheduled task"""
    try:
        # For now, we'll create a dummy function
        async def dummy_task():
            print(f"Executing task: {request.name}")
        
        scheduler.add_interval_task(
            task_id=request.task_id,
            name=request.name,
            func=dummy_task,
            hours=request.hours,
            minutes=request.minutes,
            seconds=request.seconds,
            max_retries=request.max_retries
        )
        
        return {
            "status": "created",
            "task": scheduler.get_task_status(request.task_id)
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    """Delete a scheduled task"""
    if task_id not in scheduler.tasks:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found"
        )
    
    scheduler.remove_task(task_id)
    
    return {
        "status": "deleted",
        "message": f"Task {task_id} deleted successfully"
    }


@router.post("/tasks/{task_id}/pause")
async def pause_task(task_id: str):
    """Pause a scheduled task"""
    if task_id not in scheduler.tasks:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found"
        )
    
    scheduler.pause_task(task_id)
    
    return {
        "status": "paused",
        "message": f"Task {task_id} paused successfully"
    }


@router.post("/tasks/{task_id}/resume")
async def resume_task(task_id: str):
    """Resume a paused task"""
    if task_id not in scheduler.tasks:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found"
        )
    
    scheduler.resume_task(task_id)
    
    return {
        "status": "resumed",
        "message": f"Task {task_id} resumed successfully"
    }


@router.post("/tasks/{task_id}/run")
async def run_task_now(task_id: str):
    """Manually trigger a task to run immediately"""
    if task_id not in scheduler.tasks:
        raise HTTPException(
            status_code=404,
            detail=f"Task {task_id} not found"
        )
    
    try:
        scheduler.run_task_now(task_id)
        
        return {
            "status": "triggered",
            "message": f"Task {task_id} triggered successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to trigger task: {str(e)}"
        )
