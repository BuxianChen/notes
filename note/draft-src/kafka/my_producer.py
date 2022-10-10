from kafka import KafkaProducer
import json
import time

KAFKA_IPS = ["127.0.0.1:9092"]
kafka_producer = KafkaProducer(
    bootstrap_servers=KAFKA_IPS,
    value_serializer=lambda m: json.dumps(m).encode('utf-8')
)

topic = "test"
for i in range(10, 20):
    print(f"Iteration {i}")
    kafka_producer.send(topic, value={"a": i})
    time.sleep(0.5)
