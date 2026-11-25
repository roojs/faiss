# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Spec for the non-terminating FileSock.readline loop.

These tests fail while contrib/rpc.py FileSock.readline keeps its
end-of-line and empty-read check outside ``while True``. They pass when
that loop returns the bytes read so far on a newline or on empty input,
without reading past the newline.

The index server is started with faiss.contrib.client_server.run_index_server.
SearchServer does not override Server.exec_loop or Server.one_function, so a
connection byte reaches RestrictedUnpickler.load and then FileSock.readline.
The faiss extension is not involved in that read. When it is not importable,
extract_index_ivf is only stubbed so SearchServer.__init__ can return and the
accept thread can start; the stub is not on the socket path.

Text-protocol opcodes whose CPython 3.12 _pickle loader calls readline, and
which this server actually reaches, are covered: GLOBAL, INT, LONG, FLOAT,
STRING, UNICODE, GET, PUT, and MARK+INST. PERSID does not call readline unless
persistent_load is assigned, and a bare INST opcode fails the MARK check
before readline. Both are asserted to finish. Binary protocol 4 requests use
read(), which already stops on EOF, and are asserted to still complete.
FileSock.write and any numeric line-length cap are not part of this bug.
"""

import importlib
import os
import pickle
import socket
import sys
import threading
import time
import types
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import contrib.rpc as rpc


# Opcodes whose unpickler calls readline before any other failure.
TEXT_OPCODE_PAYLOADS = (
    ("global", b"c"),
    ("int", b"I"),
    ("long", b"L"),
    ("float", b"F"),
    ("string", b"S"),
    ("unicode", b"V"),
    ("get", b"g"),
    ("put", b"p"),
    ("mark_inst", b"(i"),
)

FINISH_TIMEOUT_S = 0.75


def _ensure_faiss_symbols():
    """Make run_index_server importable when the C extension is absent.

    SearchServer.__init__ calls extract_index_ivf before the handler thread
    is started. That call does not read the attacker socket. A missing
    extension gets a stub so the real run_index_server can start; a real
    extension is left untouched.
    """
    try:
        import faiss
    except ImportError:
        faiss = types.ModuleType("faiss")
        sys.modules["faiss"] = faiss
    if not hasattr(faiss, "Index"):
        faiss.Index = object
    if not hasattr(faiss, "extract_index_ivf"):
        faiss.extract_index_ivf = lambda index: index
    return faiss


def _make_index(faiss):
    if hasattr(faiss, "IndexFlat"):
        return faiss.IndexFlat(4)
    return object()


def _load_client_server():
    faiss = _ensure_faiss_symbols()
    client_server = importlib.import_module("contrib.client_server")
    return client_server, _make_index(faiss)


def _cpu_seconds(tid):
    clk = os.sysconf("SC_CLK_TCK")
    with open(f"/proc/self/task/{tid}/stat", "r", encoding="ascii") as stat_file:
        stat = stat_file.read()
    fields = stat[stat.rfind(")") + 2 :].split()
    # utime and stime are the 14th and 15th /proc stat fields.
    return (int(fields[11]) + int(fields[12])) / clk


def _live_tids():
    live = set()
    for tid in os.listdir("/proc/self/task"):
        if os.path.exists(f"/proc/self/task/{tid}"):
            live.add(tid)
    return live


class RpcReadlineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        client_server, index = _load_client_server()
        cls.client_server = client_server
        cls.index = index
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.bind(("127.0.0.1", 0))
        cls.port = probe.getsockname()[1]
        probe.close()
        cls._server = threading.Thread(
            target=client_server.run_index_server,
            args=(index, cls.port),
            kwargs={"v6": False},
            daemon=True,
        )
        cls._server.start()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            try:
                sock = socket.create_connection(("127.0.0.1", cls.port), timeout=0.2)
            except OSError:
                time.sleep(0.02)
                continue
            sock.close()
            time.sleep(0.05)
            return
        raise RuntimeError("run_index_server did not accept a connection")

    def finish(self, fn, timeout=FINISH_TIMEOUT_S):
        """Run fn. Raise its exception, or fail if it is still running."""
        box = {}

        def runner():
            try:
                box["value"] = fn()
            except BaseException as exc:
                box["error"] = exc

        worker = threading.Thread(target=runner, daemon=True)
        worker.start()
        worker.join(timeout)
        if worker.is_alive():
            self.fail(
                f"call did not finish within {timeout}s; "
                "FileSock.readline is still looping"
            )
        if "error" in box:
            raise box["error"]
        return box.get("value")

    def assert_handler_threads_exit(self, send_payloads, timeout=FINISH_TIMEOUT_S):
        """Each payload is one connection. Every handler thread must exit."""
        before = _live_tids()
        for payload in send_payloads:
            sock = socket.create_connection(("127.0.0.1", self.port), timeout=2)
            if payload:
                sock.sendall(payload)
            sock.close()
        # Wait out the whole window. A stuck handler is still alive at the
        # deadline; a handler that returned is gone. Returning early when no
        # new thread has been sampled yet would miss a thread that starts
        # spinning a moment after accept.
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(0.02)
        alive = _live_tids() - before
        if not alive:
            return
        cpus = {}
        for tid in alive:
            try:
                cpus[tid] = round(_cpu_seconds(tid), 3)
            except (FileNotFoundError, ProcessLookupError, OSError):
                cpus[tid] = "exited"
        self.fail(
            "index-server handler still running after %.2fs for %r; "
            "thread cpu_seconds=%s" % (timeout, send_payloads, cpus)
        )

    def test_readline_returns_each_line_and_stops_at_newline(self):
        local, peer = socket.socketpair()
        self.addCleanup(local.close)
        self.addCleanup(peer.close)
        peer.sendall(b"one\ntwo\n")
        peer.shutdown(socket.SHUT_WR)
        file_sock = rpc.FileSock(local)
        self.assertEqual(self.finish(file_sock.readline), b"one\n")
        self.assertEqual(self.finish(file_sock.readline), b"two\n")
        self.assertEqual(self.finish(file_sock.readline), b"")

    def test_readline_returns_line_while_peer_stays_open(self):
        local, peer = socket.socketpair()
        self.addCleanup(local.close)
        self.addCleanup(peer.close)
        file_sock = rpc.FileSock(local)

        def read_line():
            return file_sock.readline()

        holder = {}

        def runner():
            try:
                holder["value"] = read_line()
            except BaseException as exc:
                holder["error"] = exc

        worker = threading.Thread(target=runner, daemon=True)
        worker.start()
        time.sleep(0.05)
        peer.sendall(b"abc\nDEF")
        worker.join(FINISH_TIMEOUT_S)
        self.assertFalse(
            worker.is_alive(),
            "readline did not return after a newline while the peer stayed open",
        )
        if "error" in holder:
            raise holder["error"]
        self.assertEqual(holder["value"], b"abc\n")
        local.settimeout(FINISH_TIMEOUT_S)
        self.assertEqual(local.recv(8), b"DEF")

    def test_readline_returns_partial_line_when_peer_closes(self):
        local, peer = socket.socketpair()
        self.addCleanup(local.close)
        self.addCleanup(peer.close)
        file_sock = rpc.FileSock(local)

        def read_after_close():
            peer.sendall(b"partial")
            peer.close()
            return file_sock.readline()

        self.assertEqual(self.finish(read_after_close), b"partial")

    def test_readline_returns_empty_when_peer_already_closed(self):
        local, peer = socket.socketpair()
        self.addCleanup(local.close)
        peer.close()
        file_sock = rpc.FileSock(local)
        self.assertEqual(self.finish(file_sock.readline), b"")

    def test_protocol4_request_is_answered(self):
        sock = socket.create_connection(("127.0.0.1", self.port), timeout=2)
        self.addCleanup(sock.close)
        sock.sendall(pickle.dumps(("ping", ()), protocol=4))
        sock.settimeout(2)
        data = sock.recv(65536)
        loaded = pickle.loads(data)
        self.assertIsInstance(loaded, tuple)
        self.assertEqual(len(loaded), 2)

    def test_immediate_close_exits(self):
        self.assert_handler_threads_exit((b"",))

    def test_persid_is_rejected_without_hanging(self):
        local, peer = socket.socketpair()
        self.addCleanup(local.close)
        self.addCleanup(peer.close)
        peer.sendall(b"P")
        peer.close()
        file_sock = rpc.FileSock(local)
        with self.assertRaises(pickle.UnpicklingError):
            self.finish(lambda: rpc.RestrictedUnpickler(file_sock).load())

    def test_inst_without_mark_is_rejected_without_hanging(self):
        local, peer = socket.socketpair()
        self.addCleanup(local.close)
        self.addCleanup(peer.close)
        peer.sendall(b"i")
        peer.close()
        file_sock = rpc.FileSock(local)
        with self.assertRaises(pickle.UnpicklingError):
            self.finish(lambda: rpc.RestrictedUnpickler(file_sock).load())

    def test_truncated_text_opcodes_raise_on_the_socket(self):
        for name, payload in TEXT_OPCODE_PAYLOADS:
            with self.subTest(opcode=name):
                local, peer = socket.socketpair()
                self.addCleanup(local.close)
                self.addCleanup(peer.close)
                peer.sendall(payload)
                peer.close()
                file_sock = rpc.FileSock(local)
                with self.assertRaises(pickle.UnpicklingError):
                    self.finish(lambda: rpc.RestrictedUnpickler(file_sock).load())

    def test_complete_global_line_does_not_hang(self):
        # Two newline-terminated fields, then EOF. find_class rejects the
        # module; the call must still return that error instead of spinning.
        local, peer = socket.socketpair()
        self.addCleanup(local.close)
        self.addCleanup(peer.close)
        peer.sendall(b"cos\nsystem\n")
        peer.close()
        file_sock = rpc.FileSock(local)
        with self.assertRaises(pickle.UnpicklingError):
            self.finish(lambda: rpc.RestrictedUnpickler(file_sock).load())

    def test_index_server_threads_exit_after_text_opcodes(self):
        for name, payload in TEXT_OPCODE_PAYLOADS:
            with self.subTest(opcode=name):
                self.assert_handler_threads_exit((payload,))

    def test_index_server_threads_exit_after_newline_terminated_global(self):
        self.assert_handler_threads_exit((b"cos\nsystem\n",))

    def test_several_closed_global_connections_all_exit(self):
        self.assert_handler_threads_exit((b"c", b"c", b"c"))

    def test_client_method_call_rejects_text_opcode_reply(self):
        for name, payload in TEXT_OPCODE_PAYLOADS:
            with self.subTest(opcode=name):
                listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind(("127.0.0.1", 0))
                listener.listen(1)
                port = listener.getsockname()[1]

                def serve(listener=listener, payload=payload):
                    conn, _addr = listener.accept()
                    conn.settimeout(2)
                    try:
                        conn.recv(65536)
                    except OSError:
                        pass
                    conn.sendall(payload)
                    conn.close()
                    listener.close()

                threading.Thread(target=serve, daemon=True).start()
                client = rpc.Client("127.0.0.1", port, v6=False)
                with self.assertRaises(pickle.UnpicklingError):
                    self.finish(client.ping)


if __name__ == "__main__":
    unittest.main()
