"""
后台任务管理器：把训练/抽帧/去重等耗时操作跑在后台线程里，
把 stdout/stderr 追加到内存日志，前端轮询读取。

避免模型依赖在 import 阶段加载（根据 100016986 经验）。
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
from typing import Callable


MAX_LINES = 2000
TASKS: dict[str, "Task"] = {}  # task_id -> Task


@dataclass
class Task:
    task_id: str
    name: str
    status: str = "running"   # running / success / failed
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    log: deque[str] = field(default_factory=lambda: deque(maxlen=MAX_LINES))
    meta: dict = field(default_factory=dict)  # 清理/扫描结果等扩展信息

    def append(self, line: str):
        # 避免空行堆积
        line = line.rstrip()
        if line:
            self.log.append(line)

    def snapshot(self, from_index: int = 0) -> dict:
        lines = list(self.log)
        total = len(lines)
        return {
            "task_id": self.task_id,
            "name": self.name,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "lines_total": total,
            "lines_from": from_index if from_index < total else 0,
            "lines": lines[from_index:] if from_index < total else [],
            "elapsed_s": round((self.ended_at or time.time()) - self.started_at, 2),
            "meta": dict(self.meta),
        }


def list_tasks() -> list[dict]:
    return [t.snapshot() for t in sorted(TASKS.values(), key=lambda x: -x.started_at)]


def get_task(task_id: str) -> Task | None:
    return TASKS.get(task_id)


def _start_thread(fn: Callable, *args, **kwargs) -> threading.Thread:
    t = threading.Thread(target=fn, args=args, kwargs=kwargs, daemon=True)
    t.start()
    return t


def run_command(task_id: str, cmd: list[str] | str, cwd: str | None = None, env: dict | None = None):
    """执行 shell 命令，把 stdout/stderr 持续写入对应 Task 的 log。"""
    task = TASKS[task_id]

    merged_env = dict(os.environ)
    if env:
        merged_env.update(env)
    # 保证 Python 输出不缓冲、UTF-8
    merged_env.setdefault("PYTHONUNBUFFERED", "1")
    merged_env.setdefault("PYTHONIOENCODING", "utf-8")

    if isinstance(cmd, str):
        use_shell = True
    else:
        use_shell = False

    try:
        proc = subprocess.Popen(
            cmd,
            shell=use_shell,
            cwd=cwd,
            env=merged_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            encoding="utf-8",
            errors="replace",
        )
        task.append(f"[run] $ {cmd if isinstance(cmd, str) else ' '.join(cmd)}")
        assert proc.stdout is not None
        for line in proc.stdout:
            task.append(line)
        ret = proc.wait()
        if ret == 0:
            task.status = "success"
            task.append(f"\n[ok] 退出码 {ret}")
        else:
            task.status = "failed"
            task.append(f"\n[error] 退出码 {ret}")
    except Exception as e:  # noqa
        task.status = "failed"
        task.append(f"\n[error] {type(e).__name__}: {e}")
    finally:
        task.ended_at = time.time()


def new_task(name: str, target: Callable | None = None,
             cmd: list[str] | str | None = None,
             cwd: str | None = None, env: dict | None = None) -> str:
    """创建一个后台任务。要么 target 是 python 可调用对象，要么 cmd 是 shell 命令。"""
    task_id = uuid.uuid4().hex[:10]
    TASKS[task_id] = Task(task_id=task_id, name=name)
    if cmd is not None:
        _start_thread(run_command, task_id, cmd, cwd, env)
    elif target is not None:
        def wrapped():
            t = TASKS[task_id]
            # 把 print 重定向到 task.log
            old_stdout = sys.stdout
            buf = _LogBuffer(t.append)
            sys.stdout = buf
            sys.stderr = buf
            try:
                target()
                t.status = "success"
                t.append("\n[ok] done")
            except Exception as e:  # noqa
                t.status = "failed"
                t.append(f"\n[error] {type(e).__name__}: {e}")
                import traceback
                t.append(traceback.format_exc())
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stdout
                t.ended_at = time.time()
        _start_thread(wrapped)
    return task_id


class _LogBuffer:
    def __init__(self, sink: Callable[[str], None]):
        self.sink = sink
        self._buf = ""

    def write(self, data: str):
        self._buf += data
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self.sink(line)

    def flush(self):
        if self._buf:
            self.sink(self._buf)
            self._buf = ""
