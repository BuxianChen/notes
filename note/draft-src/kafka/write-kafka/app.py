from fastapi import FastAPI
from pydantic import BaseModel
import time
import uvicorn
from kafka import KafkaProducer
import json
from thread_helper import ThreadWithReturnValue

class Input(BaseModel):
    text: str

app = FastAPI()

KAFKA_IPS = ["127.0.0.1:9092"]
kafka_producer = KafkaProducer(
    bootstrap_servers=KAFKA_IPS,
    value_serializer=lambda m: json.dumps(m).encode('utf-8')
)

def write_to_kafka(data):
    time.sleep(1)  # 假设写数据会耗时
    kafka_producer.send("test", value=data)
    print(f"send data: {data['text']} to kafka")

@app.post("/test01")
def test01(q: Input):
    t1 = time.time()

    # 正常返回后, 此线程仍会继续运行
    # 但存疑的是: 如果此线程不及时结束, 假设调用了很多次post请求时, 写kafka的线程会过多
    # 可能的解决方案是用线程池来专门处理写入kafka
    thread = ThreadWithReturnValue(target=write_to_kafka, args=(q.dict(),))
    thread.start()

    time.sleep(0.1)
    t2 = time.time()
    t = t2 - t1
    return {"route": "/test01", "time": t, "input": q.text}

@app.post("/test02")
def test02(q: Input):
    t1 = time.time()
    # 正常返回后, 此线程仍会继续运行
    # 但存疑的是: 如果此线程不及时结束, 假设调用了很多次post请求时, 写kafka的线程会过多
    # 可能的解决方案是用线程池来专门处理写入kafka
    thread = ThreadWithReturnValue(target=write_to_kafka, args=(q.dict(),))
    thread.start()

    time.sleep(0.5)
    t2 = time.time()
    t = t2 - t1
    return {"route": "/test02", "time": t, "input": q.text}

uvicorn.run(app=app, host="0.0.0.0", port=7654)
