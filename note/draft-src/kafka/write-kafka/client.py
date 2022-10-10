import threading
import requests
from thread_helper import ThreadWithReturnValue
import time

def call01(i):
    res = requests.post(
        "http://127.0.0.1:7654/test01",
        headers={"Content-Type": "application/json"},
        json={
            "text": str(i)
        }
    )
    return res.json()

def call02(i):
    res = requests.post(
        "http://127.0.0.1:7654/test02",
        headers={"Content-Type": "application/json"},
        json={
            "text": str(i)
        }
    )
    return res.json()


threads = []
t1 = time.time()
for i in range(1):
    threads.append(ThreadWithReturnValue(target=call01, args=(i,)))
    threads.append(ThreadWithReturnValue(target=call02, args=(i,)))

for t in threads:
    t.start()

start = time.time()
for i, t in enumerate(threads):
    t.join()
    print(f"{i}_th thread waiting time: {time.time() - start: .2f}")
    start = time.time()

t2 = time.time()
print(f"total time {t2 - t1:.2f}")