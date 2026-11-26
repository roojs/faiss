# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unauthenticated process-listing disclosure on the contrib RPC server.

These tests are the spec for the fix. They fail while a TCP client can
name get_ps_stats and receive the host name or the process listing,
including another user's command line. They pass once that method no
longer runs for an unauthenticated caller.

Not started here, but the same inherited method is exposed by:
- benchs/distributed_ondisk/distributed_kmeans.py AssignServer
- benchs/distributed_ondisk/search_server.py MyServer
Those modules import the faiss extension and benchmark helpers, so they
are not instantiated. run_server(..., v6=True) is the same dispatcher
with an IPv6 wildcard bind and is not given its own case.
"""

import importlib.util
import os
import pickle
import pty
import socket
import subprocess
import sys
import threading
import time
import types
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
MARKER = "faiss_ps_marker_other_user"
# Wide columns so ps does not truncate the command field before the marker.
os.environ["COLUMNS"] = "10000"

_PS_INVOCATIONS = []
_AUDIT_ARMED = False


def _audit(event, args):
    if not _AUDIT_ARMED or event != "subprocess.Popen" or len(args) < 2:
        return
    cmd = args[1]
    text = " ".join(cmd) if isinstance(cmd, (list, tuple)) else str(cmd)
    if "ps ar" in text or "run queue" in text:
        _PS_INVOCATIONS.append(text)


sys.addaudithook(_audit)


def _load_module(name, path, package):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = package
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_rpc():
    """Load contrib/rpc.py as a real module, without an installed faiss package."""
    name = "faiss_contrib_under_test.rpc"
    if name in sys.modules:
        return sys.modules[name]
    package = "faiss_contrib_under_test"
    pkg = types.ModuleType(package)
    pkg.__path__ = [str(REPO / "contrib")]
    pkg.__package__ = package
    sys.modules[package] = pkg
    return _load_module(name, REPO / "contrib" / "rpc.py", package)


def load_client_server():
    """Load the index server. Stub faiss only when the extension is absent.

    SearchServer.__init__ calls faiss.extract_index_ivf and then never
    reads that result from get_ps_stats. The RPC dispatch under test is
    the real client_server.SearchServer and rpc.Server code.
    """
    rpc = load_rpc()
    name = "faiss_contrib_under_test.client_server"
    if name in sys.modules:
        return rpc, sys.modules[name]
    try:
        import faiss
    except ImportError:
        faiss = None
    # The repo root has a faiss/ source directory that imports as a namespace
    # package and is not the SWIG module. SearchServer only needs
    # extract_index_ivf to finish constructing; get_ps_stats never reads it.
    if faiss is None or not hasattr(faiss, "extract_index_ivf"):
        faiss_mod = types.ModuleType("faiss")

        class Index:
            pass

        faiss_mod.Index = Index
        faiss_mod.extract_index_ivf = lambda index: index
        sys.modules["faiss"] = faiss_mod
    return rpc, _load_module(
        name, REPO / "contrib" / "client_server.py", "faiss_contrib_under_test"
    )


def _free_port():
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


def _connect(port):
    deadline = time.time() + 5
    last = None
    while time.time() < deadline:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        try:
            sock.connect(("127.0.0.1", port))
            return sock
        except OSError as exc:
            last = exc
            sock.close()
            time.sleep(0.05)
    raise AssertionError(f"server did not accept a connection: {last}")


def _start_other_user_command():
    """Run a busy process as nobody, attached to a tty, so `ps a` selects it.

    Returns (visible, cleanup). visible is true when `ps ar` already shows
    that other user's command line, which is the listing get_ps_stats shells
    out to. cleanup stops the process.
    """
    master, slave = pty.openpty()
    slave_name = os.ttyname(slave)
    os.chmod(slave_name, 0o666)
    code = (
        "import os, fcntl, termios, sys\n"
        "name = sys.argv[1]\n"
        "os.setsid()\n"
        "fd = os.open(name, os.O_RDWR)\n"
        "fcntl.ioctl(fd, termios.TIOCSCTTY, 0)\n"
        "os.dup2(fd, 0); os.dup2(fd, 1); os.dup2(fd, 2)\n"
        "while True:\n"
        "    x = 1\n"
    )
    try:
        child = subprocess.Popen(
            [
                "sudo",
                "-n",
                "-u",
                "nobody",
                sys.executable,
                "-c",
                code,
                slave_name,
                MARKER,
            ],
            pass_fds=(slave,),
        )
    except OSError:
        os.close(master)
        os.close(slave)
        return False, lambda: None

    def cleanup():
        subprocess.run(
            ["sudo", "-n", "pkill", "-9", "-f", MARKER],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            child.kill()
        except OSError:
            pass
        child.wait(timeout=5)
        os.close(master)
        os.close(slave)

    visible = False
    deadline = time.time() + 3
    while time.time() < deadline:
        listed = subprocess.run(
            ["ps", "ar", "-o", "user,command"],
            text=True,
            capture_output=True,
            check=False,
        )
        if (
            listed.returncode == 0
            and MARKER in listed.stdout
            and "nobody" in listed.stdout
        ):
            visible = True
            break
        if child.poll() is not None:
            break
        time.sleep(0.05)
    return visible, cleanup


def _call_get_ps_stats(port, rpc):
    """Send the protocol-4 pickle the attack describes and read the reply."""
    sock = _connect(port)
    try:
        fs = rpc.FileSock(sock)
        pickle.dump(("get_ps_stats", ()), fs, protocol=4)
        try:
            st, ret = pickle.Unpickler(fs).load()
        except Exception:
            return ""
        return f"{st}\n{ret}"
    finally:
        sock.close()


class TestUnauthenticatedProcessListing(unittest.TestCase):
    def setUp(self):
        global _AUDIT_ARMED
        _PS_INVOCATIONS.clear()
        _AUDIT_ARMED = False
        self.visible, self.cleanup_proc = _start_other_user_command()

    def tearDown(self):
        global _AUDIT_ARMED
        _AUDIT_ARMED = False
        self.cleanup_proc()

    def _assert_not_disclosed(self, rpc, serve):
        global _AUDIT_ARMED
        port = _free_port()
        thread = threading.Thread(target=serve, args=(port,), daemon=True)
        thread.start()
        # Arm only around the attacker request so the fixture's own
        # process launch is not counted as the server shelling out.
        _AUDIT_ARMED = True
        try:
            text = _call_get_ps_stats(port, rpc)
        finally:
            _AUDIT_ARMED = False

        banner = f"============ {socket.gethostname()} uptime:"
        leaked = []
        if banner in text:
            leaked.append("the host name")
        if "run queue" in text:
            leaked.append("the process listing")
        if self.visible and MARKER in text:
            leaked.append("another user's command line")
        if _PS_INVOCATIONS:
            leaked.append("the process-listing command, which ran before any refusal")
        if leaked:
            self.fail(
                "unauthenticated caller of get_ps_stats received "
                + " and ".join(leaked)
            )

    def test_base_server_does_not_list_processes_for_anonymous_client(self):
        rpc = load_rpc()

        def serve(port):
            rpc.run_server(lambda conn: rpc.Server(conn), port)

        self._assert_not_disclosed(rpc, serve)

    def test_index_server_does_not_list_processes_for_anonymous_client(self):
        rpc, client_server = load_client_server()

        def serve(port):
            client_server.run_index_server(object(), port)

        self._assert_not_disclosed(rpc, serve)


if __name__ == "__main__":
    unittest.main()
