from fastapi import FastAPI, Query
import math
import time

app = FastAPI(title="CPU Load Demo App")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/work")
def work(iterations: int = Query(2_000_000, ge=1000, le=50_000_000)):
    start = time.time()
    x = 0.0
    for i in range(iterations):
        x += math.sqrt((i % 1000) + 1)
    duration = time.time() - start
    return {
        "iterations": iterations,
        "duration_sec": round(duration, 4),
        "result_checksum": round(x, 2)
    }
