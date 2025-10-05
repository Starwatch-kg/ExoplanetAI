"""
API endpoints for monitoring and metrics
"""

from fastapi import APIRouter, HTTPException
from typing import Optional
from datetime import datetime

from services.monitoring import MonitoringService

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])

# Global monitoring service
monitoring_service = MonitoringService()


@router.get("/dashboard")
async def get_dashboard():
    """
    Get comprehensive dashboard data
    
    Returns real-time metrics, trends, errors, and system health
    """
    return monitoring_service.get_dashboard_data()


@router.get("/metrics/realtime")
async def get_realtime_metrics():
    """Get real-time metrics from the last hour"""
    return monitoring_service.get_realtime_metrics()


@router.get("/metrics/hourly")
async def get_hourly_trends(hours: int = 24):
    """Get hourly trends for the last N hours"""
    if hours < 1 or hours > 168:  # Max 1 week
        raise HTTPException(
            status_code=400,
            detail="Hours must be between 1 and 168"
        )
    
    return {
        "period": f"last_{hours}_hours",
        "trends": monitoring_service.get_hourly_trends(hours)
    }


@router.get("/metrics/daily")
async def get_daily_trends(days: int = 7):
    """Get daily trends for the last N days"""
    if days < 1 or days > 30:
        raise HTTPException(
            status_code=400,
            detail="Days must be between 1 and 30"
        )
    
    return {
        "period": f"last_{days}_days",
        "trends": monitoring_service.get_daily_trends(days)
    }


@router.get("/health")
async def get_system_health():
    """Get current system health status"""
    return monitoring_service.get_system_health()


@router.get("/errors")
async def get_error_summary():
    """Get summary of recent errors"""
    return monitoring_service.get_error_summary()


@router.get("/report")
async def get_text_report():
    """Get a formatted text report"""
    report = monitoring_service.generate_report()
    return {
        "report": report,
        "timestamp": datetime.now().isoformat()
    }


@router.post("/export")
async def export_metrics():
    """Export metrics to JSON file"""
    filepath = monitoring_service.export_metrics()
    return {
        "status": "exported",
        "filepath": str(filepath),
        "timestamp": datetime.now().isoformat()
    }


@router.post("/record/cycle")
async def record_discovery_cycle(
    targets_processed: int,
    candidates_found: int,
    high_confidence_count: int,
    processing_time: float,
    errors: int = 0
):
    """Manually record a discovery cycle (for testing)"""
    monitoring_service.record_discovery_cycle(
        targets_processed=targets_processed,
        candidates_found=candidates_found,
        high_confidence_count=high_confidence_count,
        processing_time=processing_time,
        errors=errors
    )
    
    return {
        "status": "recorded",
        "timestamp": datetime.now().isoformat()
    }
