"""One persistent, read-only chemistry child with a parent-enforced deadline.

The child calls the unchanged chemistry/QC functions but never writes the bank.
The coordinator receives its result before joining/closing the child. An owning
parent watchdog can terminate a native-code hang without waiting for Python to
deliver SIGALRM. A timed-out child must be reaped before another is started.
"""
from __future__ import annotations

import json
import fcntl
import math
import multiprocessing as mp
from multiprocessing.reduction import DupFd
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
import traceback


def _self_identity():
    """Read-only birth identity for explicit orphan recovery; no PID-only signal."""
    identity = {"pid": os.getpid(), "ppid": os.getppid(), "uid": os.getuid()}
    try:
        import ctypes
        import struct
        library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        library.proc_pidinfo.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int]
        library.proc_pidinfo.restype = ctypes.c_int
        buffer = ctypes.create_string_buffer(136)
        if library.proc_pidinfo(os.getpid(), 3, 0, buffer, len(buffer)) == len(buffer):
            identity["start_seconds"], identity["start_microseconds"] = struct.unpack_from("=QQ", buffer.raw, 120)
    except (OSError, AttributeError):
        pass
    return identity


class NativeJobTimeout(TimeoutError):
    pass


class JobDeferred(InterruptedError):
    pass


class JobHostError(RuntimeError):
    """Infrastructure fault: stop the coordinator, do not consume more jobs."""


class ChemicalJobError(RuntimeError):
    pass


def _detach_spawn_descriptor(descriptor):
    return descriptor.detach()


class _SpawnLease:
    """Reduce while spawn is pickling, using pass-fds instead of a helper socket."""
    def __init__(self, fd):
        self.fd = fd

    def __reduce__(self):
        return _detach_spawn_descriptor, (DupFd(self.fd),)


def acquire_compute_lease(path):
    """Acquire before DB recovery/spawning; an orphan child also retains it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lease = path.open("a+")
    try:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lease.close()
        raise RuntimeError(f"Chemistry compute lease is already held; existing/orphan child recovery required: {path}")
    lease.seek(0)
    lease.truncate()
    lease.write(json.dumps({"host_pid": os.getpid(), "acquired_epoch": time.time(), "path": str(path.resolve())}) + "\n")
    lease.flush()
    return lease


def atomic_status(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class ThermalGate:
    """Same conservative cool policy, outside the chemical job's deadline."""
    def __init__(self, status_dir, rest_seconds=1.0, sustained_nominal_seconds=30.0):
        self.path = Path(status_dir) / f"yield_aware_host_{os.getpid()}.json"
        self.rest_seconds = float(rest_seconds)
        self.sustained = float(sustained_nominal_seconds)
        if not all(math.isfinite(x) and x >= 0 for x in (self.rest_seconds, self.sustained)):
            raise ValueError("Thermal rest/cooldown must be finite and nonnegative")
        self.last_finished = None

    @staticmethod
    def state():
        try:
            result = subprocess.run(["/usr/bin/osascript", "-l", "JavaScript", "-e",
                'ObjC.import("Foundation"); Number($.NSProcessInfo.processInfo.thermalState)'],
                capture_output=True, text=True, timeout=5)
            state = int(result.stdout.strip())
            return state if result.returncode == 0 and state in {0, 1, 2, 3} else None
        except (OSError, ValueError, subprocess.TimeoutExpired):
            return None

    def record(self, state, action, job_id, **extra):
        atomic_status(self.path, {"pid": os.getpid(), "updated_epoch": time.time(),
            "thermal_state": state, "action": action, "job_id": job_id,
            "max_generation_compute_children": 1, "temperature_celsius_available": False, **extra})

    def wait(self, job_id, stop_requested=None):
        started = time.monotonic()
        if self.last_finished is not None:
            time.sleep(max(0.0, self.rest_seconds - (started - self.last_finished)))
        cooling, nominal_since = False, None
        while True:
            if stop_requested and stop_requested():
                self.record(None, "deferred_by_stop_file", job_id)
                raise JobDeferred("Stop requested before chemistry started")
            state = self.state()
            now = time.monotonic()
            if state == 0:
                if not cooling:
                    self.record(state, "running_one_compute_child", job_id)
                    return now - started
                nominal_since = nominal_since if nominal_since is not None else now
                if now - nominal_since >= self.sustained:
                    self.record(state, "running_after_cooldown", job_id)
                    return now - started
            else:
                cooling, nominal_since = True, None
            self.record(state, "waiting_for_sustained_nominal", job_id)
            time.sleep(5)

    def finished(self):
        self.last_finished = time.monotonic()


def _chemistry_child(connection, context, job_function, transferred_lease):
    """Spawned child owns only its Pipe and a read-only descriptor-cache handle."""
    lease_fd = None
    try:
        if transferred_lease is not None:
            lease_fd = int(transferred_lease)
            # The duplicated open-file description retains the host's flock.
            fcntl.flock(lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if job_function is None:
            import expand_million_bank as core
            core.worker_init(context["db_path"], context["holdout"], context["quality"])
            job_function = core._run_job  # Original gates/lineage, without a Python-only alarm.
        connection.send({"kind": "ready", "pid": os.getpid(), "identity": _self_identity()})
        while True:
            message = connection.recv()
            if message.get("kind") == "close":
                return
            job = message["job"]
            try:
                import random
                import numpy as np
                random.seed(job["seed"])
                np.random.seed(job["seed"])
                result = job_function(job)
                connection.send({"kind": "result", "job_id": job["job_id"], "result": result})
            except BaseException as error:
                connection.send({"kind": "error", "job_id": job.get("job_id"),
                    "error_type": type(error).__name__, "traceback": traceback.format_exc()})
    except (EOFError, BrokenPipeError):
        return
    except BaseException:
        try:
            connection.send({"kind": "startup_error", "traceback": traceback.format_exc()})
        except (OSError, EOFError):
            pass
    finally:
        connection.close()
        if lease_fd is not None:
            os.close(lease_fd)


class NativeJobHost:
    """Serial host; only a process created by this object can be terminated."""
    def __init__(self, db_path=None, holdout=None, quality=None, *, thermal_gate=None,
                 stop_requested=None, startup_timeout=30.0, job_function=None,
                 compute_lease=None, lease_path=None, python_hash_seed=0):
        self.context = {"db_path": str(Path(db_path).resolve()) if db_path else None,
                        "holdout": holdout, "quality": quality}
        self.thermal_gate, self.stop_requested = thermal_gate, stop_requested
        self.startup_timeout, self.job_function = float(startup_timeout), job_function
        self.python_hash_seed = int(python_hash_seed)
        if not 0 <= self.python_hash_seed <= 4294967295:
            raise ValueError("Python hash seed is outside the supported fixed range")
        if not math.isfinite(self.startup_timeout) or self.startup_timeout <= 0:
            raise ValueError("Startup deadline must be finite and positive")
        self.owns_lease = compute_lease is None
        self.compute_lease = compute_lease
        if self.compute_lease is None and (lease_path or db_path):
            self.compute_lease = acquire_compute_lease(lease_path or Path(db_path).with_suffix(".chemistry.lock"))
        self.process = self.connection = self.reader = self.messages = None
        self.active_job = None
        self.last_job_seconds = self.last_thermal_wait_seconds = 0.0

    @property
    def child_pid(self):
        return self.process.pid if self.process is not None else None

    def _reader(self, connection, messages):
        try:
            while True:
                messages.put(connection.recv())
        except (EOFError, OSError) as error:
            messages.put({"kind": "pipe_closed", "error": str(error)})

    def _receive(self, deadline):
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise NativeJobTimeout("Owned chemistry child exceeded its parent-enforced deadline")
            try:
                message = self.messages.get(timeout=min(0.25, remaining))
                if time.monotonic() > deadline:
                    raise NativeJobTimeout("Owned chemistry child exceeded its parent-enforced deadline")
                return message
            except queue.Empty:
                if self.process is not None and not self.process.is_alive():
                    raise JobHostError(f"Owned chemistry child exited with status {self.process.exitcode}")

    def _start(self):
        if self.process is not None and self.process.is_alive():
            return
        if self.process is not None:
            self._discard()
        parent, child = mp.get_context("spawn").Pipe(duplex=True)
        process = mp.get_context("spawn").Process(target=_chemistry_child,
            args=(child, self.context, self.job_function,
                  _SpawnLease(self.compute_lease.fileno()) if self.compute_lease is not None else None),
            name="owned_chemistry_child")
        previous_hash_seed = os.environ.get("PYTHONHASHSEED")
        try:
            os.environ["PYTHONHASHSEED"] = str(self.python_hash_seed)
            process.start()
        except BaseException as error:
            parent.close()
            child.close()
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise JobHostError(f"Unable to spawn an owned chemistry child: {error}") from error
        finally:
            if previous_hash_seed is None:
                os.environ.pop("PYTHONHASHSEED", None)
            else:
                os.environ["PYTHONHASHSEED"] = previous_hash_seed
        child.close()
        self.process, self.connection = process, parent
        self.messages = queue.Queue()
        self.reader = threading.Thread(target=self._reader, args=(parent, self.messages), daemon=True)
        self.reader.start()
        try:
            ready = self._receive(time.monotonic() + self.startup_timeout)
            if ready.get("kind") != "ready" or ready.get("pid") != process.pid:
                raise JobHostError(f"Chemistry child failed readiness: {ready}")
            if self.compute_lease is not None:
                self.compute_lease.seek(0)
                self.compute_lease.truncate()
                self.compute_lease.write(json.dumps({"host_pid": os.getpid(), "child_pid": process.pid,
                    "child_identity": ready["identity"], "ready_epoch": time.time(),
                    "recovery_policy": "lease held across parent death; refuse second compute child; verify birth identity before manual recovery"}) + "\n")
                self.compute_lease.flush()
        except BaseException as error:
            self._discard()
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise JobHostError(f"Chemistry child could not start safely: {error}") from error

    def _discard(self):
        """Terminate/reap only the current owned process, before any replacement."""
        process, connection, reader = self.process, self.connection, self.reader
        if process is not None:
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)
            if process.is_alive():
                raise JobHostError("Unable to reap owned chemistry child; refusing a second compute child")
            process.join(timeout=0)
        if connection is not None:
            connection.close()
        if reader is not None:
            reader.join(timeout=1.0)
        self.process = self.connection = self.reader = self.messages = None

    def run(self, job):
        if self.active_job is not None:
            raise RuntimeError("One host permits only one active chemistry job")
        self.active_job = job["job_id"]
        thermal_wait, started = 0.0, None
        self.last_job_seconds = self.last_thermal_wait_seconds = 0.0
        try:
            budget = float(job.get("timeout_seconds", 90))
            if not math.isfinite(budget) or budget <= 0:
                raise ValueError("Chemical deadline must be finite and positive")
            if self.thermal_gate:
                thermal_wait = self.thermal_gate.wait(job["job_id"], self.stop_requested)
            elif self.stop_requested and self.stop_requested():
                raise JobDeferred("Stop requested before chemistry started")
            self._start()
            started = time.monotonic()
            try:
                self.connection.send({"kind": "job", "job": job})
            except (OSError, EOFError) as error:
                self._discard()
                raise JobHostError(f"Unable to send an owned chemistry job: {error}") from error
            try:
                message = self._receive(started + budget)
            except BaseException:
                self._discard()
                raise
            if message.get("job_id") != job["job_id"]:
                self._discard()
                raise JobHostError("Chemistry Pipe returned the wrong job identity")
            if message.get("kind") == "error":
                raise ChemicalJobError(message["traceback"])
            if message.get("kind") != "result":
                self._discard()
                raise JobHostError(f"Unexpected chemistry message: {message}")
            result = message["result"]
            if result.get("job_id") != job["job_id"]:
                self._discard()
                raise JobHostError("Chemistry result identity disagrees")
            result["job_host_seconds"] = time.monotonic() - started
            result["thermal_wait_seconds"] = thermal_wait
            return result
        finally:
            self.active_job = None
            self.last_job_seconds = max(0, time.monotonic() - started) if started is not None else 0.0
            self.last_thermal_wait_seconds = thermal_wait
            if self.thermal_gate:
                self.thermal_gate.finished()

    def close(self):
        if self.process is not None and self.process.is_alive():
            try:
                self.connection.send({"kind": "close"})
                self.process.join(timeout=1.0)
            except (OSError, EOFError):
                pass
        self._discard()
        if self.owns_lease and self.compute_lease is not None:
            self.compute_lease.close()
            self.compute_lease = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
