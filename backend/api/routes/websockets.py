"""
WebSocket routes for real-time communication
Provides real-time updates for training progress, analysis status, and notifications
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Set
from uuid import uuid4

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.websockets import WebSocketState

from auth.jwt_auth import get_current_user_websocket
from core.logging import get_logger

logger = get_logger(__name__)

# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections and message broadcasting"""
    
    def __init__(self):
        # Active connections by session ID
        self.active_connections: Dict[str, WebSocket] = {}
        # User sessions by user ID
        self.user_sessions: Dict[str, Set[str]] = {}
        # Session metadata
        self.session_metadata: Dict[str, Dict] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str, user_id: Optional[str] = None):
        """Accept new WebSocket connection"""
        await websocket.accept()
        
        self.active_connections[session_id] = websocket
        self.session_metadata[session_id] = {
            "user_id": user_id,
            "connected_at": asyncio.get_event_loop().time(),
            "last_ping": asyncio.get_event_loop().time()
        }
        
        if user_id:
            if user_id not in self.user_sessions:
                self.user_sessions[user_id] = set()
            self.user_sessions[user_id].add(session_id)
        
        logger.info(f"WebSocket connected: session={session_id}, user={user_id}")
    
    def disconnect(self, session_id: str):
        """Remove WebSocket connection"""
        if session_id in self.active_connections:
            metadata = self.session_metadata.get(session_id, {})
            user_id = metadata.get("user_id")
            
            # Remove from active connections
            del self.active_connections[session_id]
            del self.session_metadata[session_id]
            
            # Remove from user sessions
            if user_id and user_id in self.user_sessions:
                self.user_sessions[user_id].discard(session_id)
                if not self.user_sessions[user_id]:
                    del self.user_sessions[user_id]
            
            logger.info(f"WebSocket disconnected: session={session_id}, user={user_id}")
    
    async def send_personal_message(self, message: dict, session_id: str):
        """Send message to specific session"""
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send message to session {session_id}: {e}")
                self.disconnect(session_id)
    
    async def send_user_message(self, message: dict, user_id: str):
        """Send message to all sessions of a user"""
        if user_id in self.user_sessions:
            sessions = list(self.user_sessions[user_id])  # Copy to avoid modification during iteration
            for session_id in sessions:
                await self.send_personal_message(message, session_id)
    
    async def broadcast(self, message: dict, exclude_sessions: Optional[Set[str]] = None):
        """Broadcast message to all connected sessions"""
        exclude_sessions = exclude_sessions or set()
        sessions = [sid for sid in self.active_connections.keys() if sid not in exclude_sessions]
        
        for session_id in sessions:
            await self.send_personal_message(message, session_id)
    
    async def ping_all(self):
        """Send ping to all connections to keep them alive"""
        ping_message = {
            "type": "ping",
            "timestamp": asyncio.get_event_loop().time()
        }
        
        sessions_to_remove = []
        for session_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(json.dumps(ping_message))
                self.session_metadata[session_id]["last_ping"] = asyncio.get_event_loop().time()
            except Exception:
                sessions_to_remove.append(session_id)
        
        # Remove failed connections
        for session_id in sessions_to_remove:
            self.disconnect(session_id)
    
    def get_connection_stats(self) -> dict:
        """Get connection statistics"""
        return {
            "total_connections": len(self.active_connections),
            "unique_users": len(self.user_sessions),
            "sessions_per_user": {
                user_id: len(sessions) 
                for user_id, sessions in self.user_sessions.items()
            }
        }

# Global connection manager
manager = ConnectionManager()

# Create router
router = APIRouter(prefix="/ws", tags=["websockets"])

@router.websocket("/connect/{session_id}")
async def websocket_endpoint(
    websocket: WebSocket, 
    session_id: str,
    token: Optional[str] = None
):
    """Main WebSocket endpoint for general communication"""
    user_id = None
    
    # Authenticate user if token provided
    if token:
        try:
            user = await get_current_user_websocket(token)
            user_id = user.id if user else None
        except Exception as e:
            logger.warning(f"WebSocket authentication failed: {e}")
    
    await manager.connect(websocket, session_id, user_id)
    
    try:
        # Send welcome message
        welcome_message = {
            "type": "welcome",
            "session_id": session_id,
            "user_id": user_id,
            "timestamp": asyncio.get_event_loop().time(),
            "server_info": {
                "version": "2.0.0",
                "features": ["training_progress", "analysis_updates", "notifications"]
            }
        }
        await manager.send_personal_message(welcome_message, session_id)
        
        # Listen for messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                await handle_websocket_message(message, session_id, user_id)
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                error_message = {
                    "type": "error",
                    "message": "Invalid JSON format",
                    "timestamp": asyncio.get_event_loop().time()
                }
                await manager.send_personal_message(error_message, session_id)
            except Exception as e:
                logger.error(f"WebSocket message handling error: {e}")
                error_message = {
                    "type": "error",
                    "message": "Internal server error",
                    "timestamp": asyncio.get_event_loop().time()
                }
                await manager.send_personal_message(error_message, session_id)
    
    finally:
        manager.disconnect(session_id)

@router.websocket("/training/{job_id}")
async def training_progress_websocket(
    websocket: WebSocket,
    job_id: str,
    token: Optional[str] = None
):
    """WebSocket endpoint for training progress updates"""
    session_id = f"training_{job_id}_{uuid4().hex[:8]}"
    user_id = None
    
    # Authenticate user
    if token:
        try:
            user = await get_current_user_websocket(token)
            user_id = user.id if user else None
        except Exception as e:
            await websocket.close(code=4001, reason="Authentication failed")
            return
    
    await manager.connect(websocket, session_id, user_id)
    
    try:
        # Send initial status
        initial_message = {
            "type": "training_started",
            "job_id": job_id,
            "session_id": session_id,
            "timestamp": asyncio.get_event_loop().time()
        }
        await manager.send_personal_message(initial_message, session_id)
        
        # Keep connection alive and listen for updates
        while True:
            try:
                # Check if training job still exists and is active
                # This would integrate with your ML training system
                await asyncio.sleep(1)  # Prevent busy loop
                
                # Example: Send periodic updates (replace with actual training progress)
                # progress_update = await get_training_progress(job_id)
                # if progress_update:
                #     await manager.send_personal_message(progress_update, session_id)
                
            except WebSocketDisconnect:
                break
    
    finally:
        manager.disconnect(session_id)

@router.websocket("/analysis/{target_id}")
async def analysis_progress_websocket(
    websocket: WebSocket,
    target_id: str,
    token: Optional[str] = None
):
    """WebSocket endpoint for analysis progress updates"""
    session_id = f"analysis_{target_id}_{uuid4().hex[:8]}"
    user_id = None
    
    # Authenticate user
    if token:
        try:
            user = await get_current_user_websocket(token)
            user_id = user.id if user else None
        except Exception as e:
            await websocket.close(code=4001, reason="Authentication failed")
            return
    
    await manager.connect(websocket, session_id, user_id)
    
    try:
        # Send initial status
        initial_message = {
            "type": "analysis_started",
            "target_id": target_id,
            "session_id": session_id,
            "timestamp": asyncio.get_event_loop().time()
        }
        await manager.send_personal_message(initial_message, session_id)
        
        # Keep connection alive for analysis updates
        while True:
            try:
                await asyncio.sleep(1)
                
                # Example: Send analysis progress updates
                # analysis_update = await get_analysis_progress(target_id)
                # if analysis_update:
                #     await manager.send_personal_message(analysis_update, session_id)
                
            except WebSocketDisconnect:
                break
    
    finally:
        manager.disconnect(session_id)

async def handle_websocket_message(message: dict, session_id: str, user_id: Optional[str]):
    """Handle incoming WebSocket messages"""
    message_type = message.get("type")
    
    if message_type == "pong":
        # Handle pong response
        manager.session_metadata[session_id]["last_ping"] = asyncio.get_event_loop().time()
    
    elif message_type == "subscribe":
        # Handle subscription to specific events
        topics = message.get("topics", [])
        logger.info(f"Session {session_id} subscribed to topics: {topics}")
        
        # Store subscription preferences
        if session_id in manager.session_metadata:
            manager.session_metadata[session_id]["subscriptions"] = topics
    
    elif message_type == "unsubscribe":
        # Handle unsubscription
        topics = message.get("topics", [])
        logger.info(f"Session {session_id} unsubscribed from topics: {topics}")
        
        # Update subscription preferences
        if session_id in manager.session_metadata:
            current_subs = manager.session_metadata[session_id].get("subscriptions", [])
            updated_subs = [topic for topic in current_subs if topic not in topics]
            manager.session_metadata[session_id]["subscriptions"] = updated_subs
    
    else:
        logger.warning(f"Unknown message type: {message_type}")

# Utility functions for broadcasting updates
async def broadcast_training_progress(job_id: str, progress_data: dict):
    """Broadcast training progress to relevant sessions"""
    message = {
        "type": "training_progress",
        "job_id": job_id,
        "data": progress_data,
        "timestamp": asyncio.get_event_loop().time()
    }
    
    # Send to all sessions subscribed to training updates
    relevant_sessions = []
    for session_id, metadata in manager.session_metadata.items():
        subscriptions = metadata.get("subscriptions", [])
        if "training_progress" in subscriptions or f"training_{job_id}" in subscriptions:
            relevant_sessions.append(session_id)
    
    for session_id in relevant_sessions:
        await manager.send_personal_message(message, session_id)

async def broadcast_analysis_update(target_id: str, analysis_data: dict):
    """Broadcast analysis updates to relevant sessions"""
    message = {
        "type": "analysis_update",
        "target_id": target_id,
        "data": analysis_data,
        "timestamp": asyncio.get_event_loop().time()
    }
    
    # Send to all sessions subscribed to analysis updates
    relevant_sessions = []
    for session_id, metadata in manager.session_metadata.items():
        subscriptions = metadata.get("subscriptions", [])
        if "analysis_updates" in subscriptions or f"analysis_{target_id}" in subscriptions:
            relevant_sessions.append(session_id)
    
    for session_id in relevant_sessions:
        await manager.send_personal_message(message, session_id)

async def broadcast_system_notification(notification: dict, user_ids: Optional[List[str]] = None):
    """Broadcast system notifications"""
    message = {
        "type": "system_notification",
        "data": notification,
        "timestamp": asyncio.get_event_loop().time()
    }
    
    if user_ids:
        # Send to specific users
        for user_id in user_ids:
            await manager.send_user_message(message, user_id)
    else:
        # Broadcast to all connected users
        await manager.broadcast(message)

# Background task to keep connections alive
async def websocket_keepalive():
    """Background task to ping all connections periodically"""
    while True:
        try:
            await manager.ping_all()
            await asyncio.sleep(30)  # Ping every 30 seconds
        except Exception as e:
            logger.error(f"WebSocket keepalive error: {e}")
            await asyncio.sleep(5)

# Export the router and utilities
__all__ = [
    "router",
    "manager",
    "broadcast_training_progress",
    "broadcast_analysis_update", 
    "broadcast_system_notification",
    "websocket_keepalive"
]
