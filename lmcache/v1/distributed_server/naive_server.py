# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional
import asyncio
import socket
import threading
import time
import logging
import traceback

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.distributed_server.abstract_server import (  # noqa: E501
    DistributedServerInterface,
)
from lmcache.v1.lookup_server import LookupServerInterface
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.protocol import ClientMetaMessage, Constants, ServerMetaMessage
from lmcache.v1.storage_backend.storage_manager import StorageManager

logger = init_logger(__name__)

# TODO(Jiayi): Logic related to "put" and "exists" is not implemented yet.
# Need to think when it's needed.

# TODO(Jiayi): Need to make `handle_get` async as blocking get from disk
# will affect the performance. Another simpler and cleaner option is to make
# `handle_get` always blocking but make disk loading always async.

# TODO(Jiayi): Need to find a way to make the code more concise.
# For example, consider reusing code from remote cache server?


class NaiveDistributedServer(DistributedServerInterface):
    def __init__(
        self,
        storage_manager: StorageManager,
        lookup_server: LookupServerInterface,
        loop: asyncio.AbstractEventLoop,
        config: LMCacheEngineConfig,
    ):
        self.storage_manager = storage_manager
        self.lookup_server = lookup_server

        self.url = config.distributed_url
        assert self.url is not None
        host, port = self.url.split(":")
        self.host = host
        self.port = int(port)

        self.loop = loop
        self.thread = threading.Thread(target=self.loop.run_forever)
        self.thread.start()
        asyncio.run_coroutine_threadsafe(self.start(), self.loop)

        self.async_socket_lock = asyncio.Lock()

    async def handle_get(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        """
        Handle get from the peer.
        This function is blocking for now but should be non-blocking.
        """
        memory_obj = self.storage_manager.get(key)
        return memory_obj

    def receive_all_client(
        self,
        meta: ServerMetaMessage,
        client_socket: socket.socket,
    ) -> Optional[MemoryObj]:
        received = 0
        n = meta.length

        # TODO(Jiayi): Format will be used once we support
        # compressed memory format
        memory_obj = self.storage_manager.allocate(
            meta.shape,
            meta.dtype,
            meta.fmt,
        )
        if memory_obj is None:
            logger.warning("Failed to allocate memory during remote receive")
            return None

        buffer = memory_obj.byte_array
        view = memoryview(buffer)

        while received < n:
            num_bytes = client_socket.recv_into(view[received:], n - received)
            if num_bytes == 0:
                return None
            received += num_bytes

        return memory_obj

    async def issue_get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """
        Perform get from the peer.
        This function can be blocking for now.
        """
        # `url` has the format host:port
        host_and_port = self.lookup_server.lookup(key)
        if host_and_port is None:
            return None
        host, port = host_and_port

        # TODO(Jiayi): Cache the hot client sockets if possible.
        # For example, retrieving 100 chunks could create 100 the same
        # connection for 100 times.
        # However, too many live sockets could cause file descriptor exhaustion
        # (i.e., Too many open files).
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((host, port))
        logger.debug(f"Peer connection created at {host}:{port}")

        async with self.async_socket_lock:
            client_socket.sendall(
                ClientMetaMessage(
                    Constants.CLIENT_GET,
                    key,
                    0,
                    MemoryFormat(1),
                    torch.float16,
                    torch.Size([0, 0, 0, 0]),
                ).serialize()
            )

            data = client_socket.recv(ServerMetaMessage.packlength())

        meta = ServerMetaMessage.deserialize(data)
        if meta.code != Constants.SERVER_SUCCESS:
            return None

        async with self.async_socket_lock:
            memory_obj = self.receive_all_client(meta, client_socket)

        return memory_obj

    async def receive_all_server(self, reader, n):
        data = bytearray()
        while len(data) < n:
            packet = await reader.read(n - len(data))
            if not packet:
                return None  # Client disconnected
            data.extend(packet)
        return data

    async def handle_client(self, reader, writer):
        """
        Handle the client.
        """
        # Ensure we always have a file handler for deep debug if not present
        if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', '').endswith('test.txt') for h in logger.handlers):
            try:
                fh = logging.FileHandler("/ms_test2/w00917303/test/test.txt", encoding="utf-8")
                fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
                fh.setFormatter(fmt)
                logger.addHandler(fh)
            except Exception:
                # If file handler cannot be added, continue with existing handlers
                pass

        addr = writer.get_extra_info("peername")
        conn_id = f"{addr}-{id(writer)}"
        logger.info(f"[{conn_id}] Connected")

        # Simple phase tracker
        phase = "init"
        phase_start = time.perf_counter()

        def enter(p: str):
            nonlocal phase, phase_start
            phase = p
            phase_start = time.perf_counter()

        alive = True

        async def watchdog():
            while alive:
                await asyncio.sleep(5)
                try:
                    logger.debug(f"[{conn_id}] watchdog phase={phase} elapsed={time.perf_counter()-phase_start:.3f}s")
                except Exception:
                    pass

        wd_task = asyncio.create_task(watchdog())

        async def with_timeout(coro, timeout_s: float, when: str):
            try:
                return await asyncio.wait_for(coro, timeout_s)
            except asyncio.TimeoutError:
                logger.warning(f"[{conn_id}] timeout at phase={when}")
                raise

        try:
            while True:
                enter("read_header")
                header = await with_timeout(
                    self.receive_all_server(reader, ClientMetaMessage.packlength()),
                    10.0,
                    "read_header",
                )
                if not header:
                    logger.debug(f"[{conn_id}] client closed during header read")
                    break
                logger.debug(f"[{conn_id}] header read in {time.perf_counter()-phase_start:.6f}s")

                enter("deserialize_header")
                meta = ClientMetaMessage.deserialize(header)

                match meta.command:
                    case Constants.CLIENT_GET:
                        enter("handle_get")
                        t0 = time.perf_counter()
                        memory_obj = await with_timeout(self.handle_get(meta.key), 30.0, "handle_get")
                        t1 = time.perf_counter()

                        if memory_obj is not None:
                            enter("send_meta")
                            writer.write(
                                ServerMetaMessage(
                                    Constants.SERVER_SUCCESS,
                                    len(memory_obj.byte_array),
                                    memory_obj.get_memory_format(),
                                    memory_obj.get_dtype(),
                                    memory_obj.get_shape(),
                                ).serialize()
                            )
                            await with_timeout(writer.drain(), 10.0, "drain_meta")
                            t2 = time.perf_counter()

                            enter("send_payload")
                            writer.write(memory_obj.byte_array)
                            await with_timeout(writer.drain(), 30.0, "drain_payload")
                            memory_obj.ref_count_down()
                            t3 = time.perf_counter()

                            logger.info(
                                f"[{conn_id}] get ok; get={t1 - t0:.6f}s meta={t2 - t1:.6f}s data={t3 - t2:.6f}s"
                            )
                        else:
                            enter("send_fail")
                            writer.write(
                                ServerMetaMessage(
                                    Constants.SERVER_FAIL,
                                    0,
                                    MemoryFormat(1),
                                    torch.float16,
                                    torch.Size((0, 0, 0, 0)),
                                ).serialize()
                            )
                            await with_timeout(writer.drain(), 10.0, "drain_fail")
        except Exception as e:
            logger.error(f"[{conn_id}] exception at phase={phase}: {e}\n{traceback.format_exc()}")
        finally:
            alive = False
            try:
                enter("close")
                writer.close()
                await with_timeout(writer.wait_closed(), 5.0, "wait_closed")
            except Exception as e:
                logger.warning(f"[{conn_id}] close/wait_closed error at phase={phase}: {e}")
            try:
                wd_task.cancel()
            except Exception:
                pass
            logger.info(f"[{conn_id}] Disconnected at phase={phase}")

    async def start(self):
        """
        Start the server.
        """
        server = await asyncio.start_server(self.handle_client, self.host, self.port)
        addr = server.sockets[0].getsockname()
        logger.info(f"Server started at {addr}")

        async with server:
            await server.serve_forever()

    def close(self):
        """
        Close the server.
        """
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread.is_alive():
            self.thread.join()
