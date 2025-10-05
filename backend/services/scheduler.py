"""
Task Scheduler for Automated Discovery
Provides cron-like scheduling for periodic tasks
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import schedule
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ScheduledTask:
    """Represents a scheduled task"""
    task_id: str
    name: str
    func: Callable
    schedule_type: str  # 'cron' or 'interval'
    schedule_config: Dict
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    error_count: int = 0
    max_retries: int = 3


class DiscoveryScheduler:
    """
    Scheduler for automated discovery tasks
    
    Features:
    - Cron-like scheduling (e.g., "0 */6 * * *" for every 6 hours)
    - Interval-based scheduling (e.g., every 6 hours)
    - Task monitoring and error handling
    - Automatic retries on failure
    """
    
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.tasks: Dict[str, ScheduledTask] = {}
        self.is_running = False
        logger.info("DiscoveryScheduler initialized")
    
    def start(self):
        """Start the scheduler"""
        if not self.is_running:
            self.scheduler.start()
            self.is_running = True
            logger.info("🕐 Scheduler started")
    
    def stop(self):
        """Stop the scheduler"""
        if self.is_running:
            self.scheduler.shutdown()
            self.is_running = False
            logger.info("🛑 Scheduler stopped")
    
    def add_cron_task(
        self,
        task_id: str,
        name: str,
        func: Callable,
        cron_expression: str,
        max_retries: int = 3
    ):
        """
        Add a cron-based scheduled task
        
        Args:
            task_id: Unique identifier for the task
            name: Human-readable task name
            func: Async function to execute
            cron_expression: Cron expression (e.g., "0 */6 * * *")
            max_retries: Maximum retry attempts on failure
        
        Example cron expressions:
            "0 */6 * * *" - Every 6 hours
            "0 0 * * *" - Daily at midnight
            "0 0 * * 0" - Weekly on Sunday
            "*/30 * * * *" - Every 30 minutes
        """
        # Parse cron expression
        parts = cron_expression.split()
        if len(parts) != 5:
            raise ValueError("Invalid cron expression. Expected 5 parts: minute hour day month day_of_week")
        
        minute, hour, day, month, day_of_week = parts
        
        # Create trigger
        trigger = CronTrigger(
            minute=minute,
            hour=hour,
            day=day,
            month=month,
            day_of_week=day_of_week
        )
        
        # Wrap function with error handling
        async def wrapped_func():
            await self._execute_task(task_id, func)
        
        # Add to scheduler
        job = self.scheduler.add_job(
            wrapped_func,
            trigger=trigger,
            id=task_id,
            name=name,
            replace_existing=True
        )
        
        # Store task info
        self.tasks[task_id] = ScheduledTask(
            task_id=task_id,
            name=name,
            func=func,
            schedule_type="cron",
            schedule_config={"cron": cron_expression},
            next_run=job.next_run_time,
            max_retries=max_retries
        )
        
        logger.info(f"📅 Added cron task '{name}' with schedule: {cron_expression}")
    
    def add_interval_task(
        self,
        task_id: str,
        name: str,
        func: Callable,
        hours: int = 0,
        minutes: int = 0,
        seconds: int = 0,
        max_retries: int = 3
    ):
        """
        Add an interval-based scheduled task
        
        Args:
            task_id: Unique identifier for the task
            name: Human-readable task name
            func: Async function to execute
            hours: Interval in hours
            minutes: Interval in minutes
            seconds: Interval in seconds
            max_retries: Maximum retry attempts on failure
        """
        if hours == 0 and minutes == 0 and seconds == 0:
            raise ValueError("At least one time unit must be specified")
        
        # Create trigger
        trigger = IntervalTrigger(
            hours=hours,
            minutes=minutes,
            seconds=seconds
        )
        
        # Wrap function with error handling
        async def wrapped_func():
            await self._execute_task(task_id, func)
        
        # Add to scheduler
        job = self.scheduler.add_job(
            wrapped_func,
            trigger=trigger,
            id=task_id,
            name=name,
            replace_existing=True
        )
        
        # Store task info
        self.tasks[task_id] = ScheduledTask(
            task_id=task_id,
            name=name,
            func=func,
            schedule_type="interval",
            schedule_config={
                "hours": hours,
                "minutes": minutes,
                "seconds": seconds
            },
            next_run=job.next_run_time,
            max_retries=max_retries
        )
        
        interval_str = f"{hours}h {minutes}m {seconds}s"
        logger.info(f"⏱️ Added interval task '{name}' with interval: {interval_str}")
    
    async def _execute_task(self, task_id: str, func: Callable):
        """Execute a task with error handling and retries"""
        task = self.tasks.get(task_id)
        if not task:
            logger.error(f"Task {task_id} not found")
            return
        
        task.status = TaskStatus.RUNNING
        task.last_run = datetime.now()
        
        logger.info(f"▶️ Executing task: {task.name}")
        
        try:
            # Execute the function
            if asyncio.iscoroutinefunction(func):
                await func()
            else:
                func()
            
            task.status = TaskStatus.COMPLETED
            task.error_count = 0
            logger.info(f"✅ Task completed: {task.name}")
            
        except Exception as e:
            task.error_count += 1
            logger.error(f"❌ Task failed: {task.name} - {str(e)}", exc_info=True)
            
            if task.error_count >= task.max_retries:
                task.status = TaskStatus.FAILED
                logger.error(
                    f"🚫 Task {task.name} failed after {task.error_count} attempts. "
                    "Manual intervention required."
                )
            else:
                task.status = TaskStatus.PENDING
                logger.warning(
                    f"🔄 Task {task.name} will retry. "
                    f"Attempt {task.error_count}/{task.max_retries}"
                )
        
        # Update next run time
        job = self.scheduler.get_job(task_id)
        if job:
            task.next_run = job.next_run_time
    
    def remove_task(self, task_id: str):
        """Remove a scheduled task"""
        if task_id in self.tasks:
            self.scheduler.remove_job(task_id)
            del self.tasks[task_id]
            logger.info(f"🗑️ Removed task: {task_id}")
    
    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get status of a specific task"""
        task = self.tasks.get(task_id)
        if not task:
            return None
        
        return {
            "task_id": task.task_id,
            "name": task.name,
            "schedule_type": task.schedule_type,
            "schedule_config": task.schedule_config,
            "status": task.status.value,
            "last_run": task.last_run.isoformat() if task.last_run else None,
            "next_run": task.next_run.isoformat() if task.next_run else None,
            "error_count": task.error_count,
            "max_retries": task.max_retries
        }
    
    def get_all_tasks(self) -> List[Dict]:
        """Get status of all tasks"""
        return [
            self.get_task_status(task_id)
            for task_id in self.tasks.keys()
        ]
    
    def pause_task(self, task_id: str):
        """Pause a task"""
        job = self.scheduler.get_job(task_id)
        if job:
            job.pause()
            logger.info(f"⏸️ Paused task: {task_id}")
    
    def resume_task(self, task_id: str):
        """Resume a paused task"""
        job = self.scheduler.get_job(task_id)
        if job:
            job.resume()
            logger.info(f"▶️ Resumed task: {task_id}")
    
    def run_task_now(self, task_id: str):
        """Manually trigger a task to run immediately"""
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        logger.info(f"🚀 Manually triggering task: {task.name}")
        asyncio.create_task(self._execute_task(task_id, task.func))


# Example usage for automated discovery
async def setup_discovery_schedule(scheduler: DiscoveryScheduler, discovery_service):
    """
    Setup automated discovery schedule
    
    This configures the scheduler to run discovery tasks at regular intervals
    """
    
    # Main discovery cycle - every 6 hours
    scheduler.add_cron_task(
        task_id="main_discovery",
        name="Main Discovery Cycle",
        func=discovery_service._discovery_cycle,
        cron_expression="0 */6 * * *",  # Every 6 hours
        max_retries=3
    )
    
    # Model retraining - daily at 2 AM
    async def retrain_model():
        logger.info("🔄 Retraining ML model with new data...")
        # Add model retraining logic here
    
    scheduler.add_cron_task(
        task_id="model_retrain",
        name="Daily Model Retraining",
        func=retrain_model,
        cron_expression="0 2 * * *",  # Daily at 2 AM
        max_retries=2
    )
    
    # Cleanup old data - weekly on Sunday at 3 AM
    async def cleanup_old_data():
        logger.info("🧹 Cleaning up old discovery data...")
        # Add cleanup logic here
    
    scheduler.add_cron_task(
        task_id="data_cleanup",
        name="Weekly Data Cleanup",
        func=cleanup_old_data,
        cron_expression="0 3 * * 0",  # Sunday at 3 AM
        max_retries=1
    )
    
    # Health check - every 30 minutes
    async def health_check():
        logger.info("💓 Running health check...")
        stats = await discovery_service.get_statistics()
        logger.info(f"Health: {stats}")
    
    scheduler.add_interval_task(
        task_id="health_check",
        name="System Health Check",
        func=health_check,
        minutes=30,
        max_retries=1
    )
    
    logger.info("✅ Discovery schedule configured successfully")
