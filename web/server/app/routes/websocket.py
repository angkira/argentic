"""
WebSocket routes for real-time message bus communication.
"""
import socketio
from fastapi import FastAPI
from typing import Dict, Set
import asyncio
from datetime import datetime
import uuid

# Create Socket.IO server
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins='*',
    logger=True,
    engineio_logger=True
)

# Track subscriptions: topic -> set of session IDs
subscriptions: Dict[str, Set[str]] = {}

# Track connected clients
connected_clients: Set[str] = set()


@sio.event
async def connect(sid, environ):
    """Handle client connection"""
    print(f"Client connected: {sid}")
    connected_clients.add(sid)
    await sio.emit('connection_status', {'status': 'connected'}, room=sid)


@sio.event
async def disconnect(sid):
    """Handle client disconnection"""
    print(f"Client disconnected: {sid}")
    connected_clients.discard(sid)

    # Remove from all subscriptions
    for topic_subs in subscriptions.values():
        topic_subs.discard(sid)


@sio.event
async def subscribe(sid, data):
    """Subscribe to a topic"""
    topic = data.get('topic')
    if not topic:
        return

    if topic not in subscriptions:
        subscriptions[topic] = set()

    subscriptions[topic].add(sid)
    print(f"Client {sid} subscribed to topic: {topic}")
    await sio.emit('subscribed', {'topic': topic}, room=sid)


@sio.event
async def unsubscribe(sid, data):
    """Unsubscribe from a topic"""
    topic = data.get('topic')
    if not topic:
        return

    if topic in subscriptions:
        subscriptions[topic].discard(sid)
        print(f"Client {sid} unsubscribed from topic: {topic}")
        await sio.emit('unsubscribed', {'topic': topic}, room=sid)


@sio.event
async def publish(sid, data):
    """Publish a message to a topic"""
    topic = data.get('topic')
    message = data.get('message')

    if not topic or not message:
        return

    # Broadcast to all subscribers of this topic
    if topic in subscriptions:
        for subscriber_sid in subscriptions[topic]:
            await sio.emit('message_bus', message, room=subscriber_sid)


async def broadcast_message(topic: str, message: dict):
    """
    Broadcast a message to all subscribers of a topic.
    This function can be called from agent services.
    """
    # Add message metadata
    message_with_meta = {
        'id': str(uuid.uuid4()),
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'topic': topic,
        **message
    }

    # Broadcast to all connected clients (for now, can filter by topic later)
    await sio.emit('message_bus', message_with_meta)

    print(f"Broadcasted message to topic {topic}: {message_with_meta['id']}")


def create_socket_app(app: FastAPI) -> socketio.ASGIApp:
    """
    Create and configure Socket.IO ASGI app.
    Mount this to the main FastAPI app.
    """
    socket_app = socketio.ASGIApp(
        socketio_server=sio,
        other_asgi_app=app,
        socketio_path='/ws/socket.io'
    )
    return socket_app


def get_sio() -> socketio.AsyncServer:
    """Get Socket.IO server instance"""
    return sio
