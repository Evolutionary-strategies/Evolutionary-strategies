import asyncio
import numpy as np

counter = 5
async def handler(reader, writer):
    x = counter
    data = await reader.read(100)
    message = data.decode()
    addr = writer.get_extra_info('peername')
    print(f"Received {message!r} from {addr!r}")
    if message == "Gimme work":
        writer.write(np.array([[0.1,0.3,0.9],[0.2,0.2,0.2]]).tobytes())
        print(f"Send: {message!r}")
    
    await writer.drain()

    print("Close the connection")
    writer.close()

async def main():
    server = await asyncio.start_server(
        handler, '127.0.0.1', 8888)

    addrs = ', '.join(str(sock.getsockname()) for sock in server.sockets)
    print(f'Serving on {addrs}')

    async with server:
        await server.serve_forever()

asyncio.run(main())