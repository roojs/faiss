# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""RPC unpickler must not run attacker-controlled code.

faiss.contrib.rpc.RestrictedUnpickler accepts every attribute of the numpy
module. A peer can therefore call numpy.savetxt and numpy.load while the
message is still being decoded. These tests send that message through the
real server and client sockets and require that neither the file write nor
the nested command runs.

They fail while the allowlist is in place.
"""

import os
import pickle
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import contrib.rpc as rpc  # noqa: E402


def _short_unicode(text):
    encoded = text.encode("utf-8")
    if len(encoded) > 255:
        raise AssertionError("pickle string does not fit SHORT_BINUNICODE")
    return b"\x8c" + bytes([len(encoded)]) + encoded


def _stack_global(module, name):
    # STACK_GLOBAL reads both names with counted binary strings, so the
    # server's broken FileSock.readline is never called.
    return _short_unicode(module) + _short_unicode(name) + b"\x93"


def _tuple(parts):
    return b"(" + b"".join(parts) + b"t"


def _command_header(marker_path):
    """Protocol-0 pickle. numpy.load falls back to pickle.load on this text."""

    class _Run(object):
        def __reduce__(self):
            return (os.system, ("touch %s" % marker_path,))

    return pickle.dumps(_Run(), protocol=0).decode("ascii")


def _chain_message(dropped_path, marker_path, result):
    """Binary pickle that calls savetxt, then load, then yields result.

    savetxt's positional arguments match numpy 1.26 and numpy 2.0:
    fname, X, fmt, delimiter, newline, header, footer, comments.
    load is called as numpy.load(path, None, True), so allow_pickle is set.
    """
    one_element_list = b"]" + b"K\x01" + b"a"
    header = _command_header(marker_path)
    savetxt_args = _tuple([
        _short_unicode(dropped_path),
        one_element_list,
        _short_unicode("%.18e"),
        _short_unicode(" "),
        _short_unicode("\n"),
        _short_unicode(header),
        _short_unicode(""),
        _short_unicode(""),
    ])
    load_args = _tuple([
        _short_unicode(dropped_path),
        b"N",
        b"\x88",
    ])
    return (
        b"\x80\x04"
        + _stack_global("numpy", "savetxt") + savetxt_args + b"R0"
        + _stack_global("numpy", "load") + load_args + b"R0"
        + result
        + b"."
    )


def _safe_leaf(path):
    # os.system goes through the shell. The temp path is chosen here and
    # must stay a single word so the marker is the file we check.
    if any(ch in path for ch in " \t\n;|&$`'\\\"<>"):
        raise AssertionError("temporary path is not a single shell word")


class RpcUnpicklerAllowlistTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="faiss-rpc-allowlist-")
        self.dropped = os.path.join(self.tmp, "dropped.txt")
        self.marker = os.path.join(self.tmp, "marker")
        _safe_leaf(self.dropped)
        _safe_leaf(self.marker)

    def tearDown(self):
        for path in (self.dropped, self.marker):
            if os.path.exists(path):
                os.remove(path)
        os.rmdir(self.tmp)

    def assert_no_unpickle_side_effects(self):
        problems = []
        if os.path.exists(self.dropped):
            problems.append(
                "numpy.savetxt wrote %s while RestrictedUnpickler.load was still running"
                % self.dropped
            )
        if os.path.exists(self.marker):
            problems.append(
                "a command from the nested pickle ran and created %s" % self.marker
            )
        self.assertFalse(problems, "; ".join(problems))

    def test_server_request_does_not_run_during_unpickle(self):
        # run_index_server is rpc.run_server(lambda s: SearchServer(s, index)).
        # SearchServer does not override one_function, which unpickles the
        # request before it looks up the method name.
        message = _chain_message(
            self.dropped,
            self.marker,
            _short_unicode("get_ntotal") + b")" + b"\x86",
        )
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()

        proc = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--serve", str(port)],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        sock = None
        try:
            deadline = time.time() + 5
            while True:
                if proc.poll() is not None:
                    err = proc.stderr.read().decode("utf-8", "replace")
                    self.fail("index server exited before accept: %s" % err)
                try:
                    sock = socket.create_connection(("127.0.0.1", port), 0.2)
                    break
                except OSError:
                    if time.time() > deadline:
                        self.fail("index server did not accept a connection")
                    time.sleep(0.05)
            sock.sendall(message)
            sock.shutdown(socket.SHUT_WR)
            sock.settimeout(0.2)
            deadline = time.time() + 5
            while time.time() < deadline:
                if os.path.exists(self.dropped) or os.path.exists(self.marker):
                    time.sleep(0.2)
                    break
                if proc.poll() is not None:
                    break
                try:
                    if not sock.recv(4096):
                        break
                except socket.timeout:
                    continue
                except OSError:
                    break
            self.assert_no_unpickle_side_effects()
        finally:
            if sock is not None:
                sock.close()
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
            else:
                proc.wait(timeout=5)

    def test_client_reply_does_not_run_during_unpickle(self):
        # ClientIndex calls rpc.Client methods, which pickle a request and
        # then decode the reply in Client.get_result with the same unpickler.
        message = _chain_message(self.dropped, self.marker, b"NN\x86")
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]

        def serve():
            conn, _addr = listener.accept()
            conn.settimeout(5)
            try:
                conn.recv(65536)
                conn.sendall(message)
                conn.shutdown(socket.SHUT_WR)
            finally:
                time.sleep(0.5)
                conn.close()

        thread = threading.Thread(target=serve)
        thread.start()
        client = None
        try:
            client = rpc.Client("127.0.0.1", port)
            try:
                client.get_ntotal()
            except Exception:
                pass
            self.assert_no_unpickle_side_effects()
        finally:
            if client is not None:
                client.sock.close()
            listener.close()
            thread.join(timeout=5)


def _serve(port):
    rpc.run_server(lambda conn: rpc.Server(conn), port)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--serve":
        _serve(int(sys.argv[2]))
    else:
        unittest.main()
