import asyncio
from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaConnectionError


async def check_kafka_connection():
    loop = asyncio.get_event_loop()
    producer = AIOKafkaProducer(
        loop=loop,
        bootstrap_servers="localhost:9092",
        client_id="kafka-check-client",
        request_timeout_ms=5000,  # 5 seconds timeout for connection attempt
    )
    try:
        print("Attempting to connect to Kafka broker at localhost:9092...")
        await producer.start()
        print("Successfully connected to Kafka broker!")
        print("Fetching metadata (this will list topics if any are auto-created or exist)...")
        metadata = await producer.client.fetch_all_metadata()
        print(f"Available topics: {metadata.topics()}")
    except KafkaConnectionError as e:
        print(f"Failed to connect to Kafka broker: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if (
            hasattr(producer, "_sender") and producer._sender.sender_task
        ):  # Check if producer was started
            print("Stopping producer...")
            await producer.stop()
            print("Producer stopped.")
        else:
            print("Producer was not started, no need to stop.")


if __name__ == "__main__":
    asyncio.run(check_kafka_connection())
