
import asyncio
from subprocess import Popen
import numpy as np



async def communicate():
    reader, writer = await asyncio.open_connection(
        '127.0.0.1', 8888)

    print(f'Send: {"Gimme work"!r}')
    writer.write("Gimme work".encode())
    await writer.drain()

    data = await reader.read(100)
    weights = np.frombuffer(data).reshape(2,3)
    writer.close()
    await writer.wait_closed()

    run_workers(weights)

def run_workers(data):
    procs = []
    for weight in data:
        procs.append(Popen(["python3", "worker.py",weight.tobytes()]))
    for p in procs:
        p.wait()
    return np.array([0,0,0,0,0])


asyncio.run(communicate())
