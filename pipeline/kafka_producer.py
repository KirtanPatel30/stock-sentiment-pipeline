"""
pipeline/kafka_producer.py
Simulates a Kafka producer streaming headlines into a topic.
Falls back to file-based queue if Kafka is not running.
"""

import json
import time
import random
from datetime import datetime
from pathlib import Path

QUEUE_FILE = Path(__file__).parent.parent / "data" / "kafka_queue.jsonl"


def produce_to_kafka(topic="stock-headlines", broker="localhost:9092"):
    """Try to produce to real Kafka, fallback to file queue."""
    try:
        from kafka import KafkaProducer
        producer = KafkaProducer(
            bootstrap_servers=[broker],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            request_timeout_ms=5000,
        )
        print(f"[KAFKA] Connected to broker {broker}")
        return producer, True
    except Exception as e:
        print(f"[KAFKA] Kafka unavailable ({e}). Using file-based queue.")
        return None, False


def stream_headlines(n=100, delay=0.05):
    """Stream n headlines to Kafka topic (or file queue as fallback)."""
    from data.fetch_news import generate_mock_headlines
    df = generate_mock_headlines(n=n)

    producer, is_kafka = produce_to_kafka()
    QUEUE_FILE.parent.mkdir(parents=True, exist_ok=True)

    sent = 0
    for _, row in df.iterrows():
        msg = {
            "id":        row["id"],
            "ticker":    row["ticker"],
            "text":      row["text"],
            "source":    row["source"],
            "score":     int(row["score"]),
            "timestamp": row["timestamp"],
        }
        if is_kafka:
            producer.send("stock-headlines", value=msg)
        else:
            with open(QUEUE_FILE, "a") as f:
                f.write(json.dumps(msg) + "\n")
        sent += 1
        time.sleep(delay)

    if is_kafka:
        producer.flush()
        print(f"[KAFKA] Sent {sent} messages to topic stock-headlines")
    else:
        print(f"[KAFKA] Wrote {sent} messages to file queue at {QUEUE_FILE}")


if __name__ == "__main__":
    stream_headlines(n=50, delay=0)
