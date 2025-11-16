"""Tests for WebSocket functionality."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from app.routes.websocket import (
    sio,
    connect,
    disconnect,
    subscribe,
    unsubscribe,
    publish,
    broadcast_message,
    connected_clients,
    subscriptions,
)


@pytest.mark.asyncio
async def test_connect_event():
    """Test client connection event."""
    # Clear connected clients
    connected_clients.clear()

    # Mock emit
    with patch.object(sio, 'emit', new_callable=AsyncMock) as mock_emit:
        sid = 'test-session-id'
        environ = {}

        await connect(sid, environ)

        # Check client was added to connected set
        assert sid in connected_clients

        # Check connection status was emitted
        mock_emit.assert_called_once_with(
            'connection_status',
            {'status': 'connected'},
            room=sid
        )


@pytest.mark.asyncio
async def test_disconnect_event():
    """Test client disconnection event."""
    # Setup
    connected_clients.clear()
    subscriptions.clear()
    sid = 'test-session-id'
    topic = 'test-topic'

    # Add client to connected set and subscriptions
    connected_clients.add(sid)
    subscriptions[topic] = {sid}

    await disconnect(sid)

    # Check client was removed from connected set
    assert sid not in connected_clients

    # Check client was removed from subscriptions
    assert sid not in subscriptions[topic]


@pytest.mark.asyncio
async def test_subscribe_event():
    """Test topic subscription."""
    subscriptions.clear()

    with patch.object(sio, 'emit', new_callable=AsyncMock) as mock_emit:
        sid = 'test-session-id'
        topic = 'test-topic'
        data = {'topic': topic}

        await subscribe(sid, data)

        # Check subscription was added
        assert topic in subscriptions
        assert sid in subscriptions[topic]

        # Check subscribed confirmation was emitted
        mock_emit.assert_called_once_with(
            'subscribed',
            {'topic': topic},
            room=sid
        )


@pytest.mark.asyncio
async def test_subscribe_without_topic():
    """Test subscription without topic does nothing."""
    subscriptions.clear()

    with patch.object(sio, 'emit', new_callable=AsyncMock) as mock_emit:
        sid = 'test-session-id'
        data = {}  # No topic

        await subscribe(sid, data)

        # Check no subscription was added
        assert len(subscriptions) == 0

        # Check no emit was called
        mock_emit.assert_not_called()


@pytest.mark.asyncio
async def test_unsubscribe_event():
    """Test topic unsubscription."""
    subscriptions.clear()
    sid = 'test-session-id'
    topic = 'test-topic'

    # Setup subscription
    subscriptions[topic] = {sid}

    with patch.object(sio, 'emit', new_callable=AsyncMock) as mock_emit:
        data = {'topic': topic}

        await unsubscribe(sid, data)

        # Check subscription was removed
        assert sid not in subscriptions[topic]

        # Check unsubscribed confirmation was emitted
        mock_emit.assert_called_once_with(
            'unsubscribed',
            {'topic': topic},
            room=sid
        )


@pytest.mark.asyncio
async def test_unsubscribe_without_topic():
    """Test unsubscription without topic does nothing."""
    subscriptions.clear()

    with patch.object(sio, 'emit', new_callable=AsyncMock) as mock_emit:
        sid = 'test-session-id'
        data = {}  # No topic

        await unsubscribe(sid, data)

        # Check no emit was called
        mock_emit.assert_not_called()


@pytest.mark.asyncio
async def test_publish_event():
    """Test message publishing to subscribers."""
    subscriptions.clear()
    topic = 'test-topic'
    publisher_sid = 'publisher-sid'
    subscriber_sid_1 = 'subscriber-1'
    subscriber_sid_2 = 'subscriber-2'

    # Setup subscriptions
    subscriptions[topic] = {subscriber_sid_1, subscriber_sid_2}

    with patch.object(sio, 'emit', new_callable=AsyncMock) as mock_emit:
        message = {
            'agent_id': 'agent-1',
            'agent_role': 'test_agent',
            'message_type': 'event',
            'content': {'test': 'data'}
        }
        data = {'topic': topic, 'message': message}

        await publish(publisher_sid, data)

        # Check message was emitted to all subscribers
        assert mock_emit.call_count == 2
        calls = mock_emit.call_args_list
        emitted_sids = {call[1]['room'] for call in calls}
        assert emitted_sids == {subscriber_sid_1, subscriber_sid_2}


@pytest.mark.asyncio
async def test_publish_without_topic():
    """Test publishing without topic does nothing."""
    with patch.object(sio, 'emit', new_callable=AsyncMock) as mock_emit:
        sid = 'test-session-id'
        data = {'message': {'test': 'data'}}  # No topic

        await publish(sid, data)

        # Check no emit was called
        mock_emit.assert_not_called()


@pytest.mark.asyncio
async def test_publish_without_message():
    """Test publishing without message does nothing."""
    with patch.object(sio, 'emit', new_callable=AsyncMock) as mock_emit:
        sid = 'test-session-id'
        data = {'topic': 'test-topic'}  # No message

        await publish(sid, data)

        # Check no emit was called
        mock_emit.assert_not_called()


@pytest.mark.asyncio
async def test_broadcast_message():
    """Test broadcasting message with metadata."""
    with patch.object(sio, 'emit', new_callable=AsyncMock) as mock_emit:
        topic = 'agent.lifecycle'
        message = {
            'agent_id': 'agent-1',
            'agent_role': 'test_agent',
            'message_type': 'event',
            'content': {'status': 'started'}
        }

        await broadcast_message(topic, message)

        # Check message was emitted
        mock_emit.assert_called_once()
        call_args = mock_emit.call_args

        # Check event name
        assert call_args[0][0] == 'message_bus'

        # Check message structure
        emitted_message = call_args[0][1]
        assert 'id' in emitted_message
        assert 'timestamp' in emitted_message
        assert emitted_message['topic'] == topic
        assert emitted_message['agent_id'] == 'agent-1'
        assert emitted_message['agent_role'] == 'test_agent'
        assert emitted_message['message_type'] == 'event'
        assert emitted_message['content'] == {'status': 'started'}


@pytest.mark.asyncio
async def test_multiple_subscribers_same_topic():
    """Test multiple clients can subscribe to same topic."""
    subscriptions.clear()
    topic = 'test-topic'
    sid1 = 'client-1'
    sid2 = 'client-2'
    sid3 = 'client-3'

    with patch.object(sio, 'emit', new_callable=AsyncMock):
        await subscribe(sid1, {'topic': topic})
        await subscribe(sid2, {'topic': topic})
        await subscribe(sid3, {'topic': topic})

        # Check all clients are subscribed
        assert topic in subscriptions
        assert len(subscriptions[topic]) == 3
        assert sid1 in subscriptions[topic]
        assert sid2 in subscriptions[topic]
        assert sid3 in subscriptions[topic]


@pytest.mark.asyncio
async def test_client_multiple_topics():
    """Test single client can subscribe to multiple topics."""
    subscriptions.clear()
    sid = 'test-client'
    topic1 = 'topic-1'
    topic2 = 'topic-2'
    topic3 = 'topic-3'

    with patch.object(sio, 'emit', new_callable=AsyncMock):
        await subscribe(sid, {'topic': topic1})
        await subscribe(sid, {'topic': topic2})
        await subscribe(sid, {'topic': topic3})

        # Check client is in all topic subscriptions
        assert sid in subscriptions[topic1]
        assert sid in subscriptions[topic2]
        assert sid in subscriptions[topic3]


@pytest.mark.asyncio
async def test_disconnect_removes_from_all_subscriptions():
    """Test disconnecting removes client from all topic subscriptions."""
    subscriptions.clear()
    connected_clients.clear()
    sid = 'test-client'
    topic1 = 'topic-1'
    topic2 = 'topic-2'

    # Setup subscriptions
    connected_clients.add(sid)
    subscriptions[topic1] = {sid}
    subscriptions[topic2] = {sid}

    await disconnect(sid)

    # Check client removed from all topics
    assert sid not in subscriptions[topic1]
    assert sid not in subscriptions[topic2]
    assert sid not in connected_clients
