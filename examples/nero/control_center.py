from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[2]
UI_FILE = Path(__file__).with_name("control_center.html")
LOG_LIMIT = 3000
PROFILE_VERSIONS = {
    "v111": "1.11",
    "v112": "1.12",
    "v120": "1.20",
    "v121": "1.21",
}
ALLOWED_TASKS = {"read_only", "teleoperate", "disable"}


class ConfigError(ValueError):
    pass


def _channel(value: Any) -> str:
    text = str(value).strip()
    if not re.fullmatch(r"\d{1,2}", text) or not 0 <= int(text) <= 31:
        raise ConfigError("CAN 通道必须是 0 到 31 的整数")
    return str(int(text))


def _number(value: Any, name: str, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} 必须是数字") from exc
    if not low <= number <= high:
        raise ConfigError(f"{name} 必须在 {low:g} 到 {high:g} 之间")
    return number


def normalize_config(payload: dict[str, Any], task: str) -> dict[str, Any]:
    if task not in ALLOWED_TASKS:
        raise ConfigError("未知操作")
    leader_channel = _channel(payload.get("leader_channel"))
    follower_channel = _channel(payload.get("follower_channel"))
    if leader_channel == follower_channel:
        raise ConfigError("主臂和从臂不能使用同一个 CAN 通道")

    leader_firmware = str(payload.get("leader_firmware", "")).lower()
    follower_firmware = str(payload.get("follower_firmware", "")).lower()
    if leader_firmware not in PROFILE_VERSIONS or follower_firmware not in PROFILE_VERSIONS:
        raise ConfigError("主臂和从臂都必须选择明确的固件档位")

    config: dict[str, Any] = {
        "leader_channel": leader_channel,
        "follower_channel": follower_channel,
        "leader_firmware": leader_firmware,
        "follower_firmware": follower_firmware,
        "leader_version": PROFILE_VERSIONS[leader_firmware],
        "follower_version": PROFILE_VERSIONS[follower_firmware],
    }

    if task in {"teleoperate", "disable"}:
        if payload.get("safety_confirmed") is not True:
            raise ConfigError("执行会改变硬件状态前必须逐项确认实体安全条件")

    if task == "teleoperate":
        config.update(
            fps=int(_number(payload.get("fps", 30), "发送频率", 1, 60)),
            speed_percent=int(_number(payload.get("speed_percent", 20), "控制器速度", 1, 30)),
            gripper_force=_number(payload.get("gripper_force", 1.0), "夹持力", 0.1, 10.0),
            max_joint_speed=_number(payload.get("max_joint_speed", 0.3), "关节速度限制", 0.05, 0.52),
            mapping=str(payload.get("mapping", "relative")),
            return_to_start=bool(payload.get("return_to_start", False)),
        )
        if config["mapping"] not in {"relative", "absolute"}:
            raise ConfigError("映射模式必须是 relative 或 absolute")
        raw_duration = payload.get("duration")
        if raw_duration in (None, ""):
            config["duration"] = None
        else:
            config["duration"] = _number(raw_duration, "运行时长", 0.5, 3600.0)

    return config


def config_signature(config: dict[str, Any]) -> str:
    return "|".join(
        config[key]
        for key in (
            "leader_channel",
            "follower_channel",
            "leader_firmware",
            "follower_firmware",
            "leader_version",
            "follower_version",
        )
    )


def build_command(task: str, config: dict[str, Any], python: str | None = None) -> list[str]:
    python = python or sys.executable
    common = [
        "--leader-channel", config["leader_channel"],
        "--follower-channel", config["follower_channel"],
        "--leader-firmware", config["leader_firmware"],
        "--follower-firmware", config["follower_firmware"],
        "--leader-version", config["leader_version"],
        "--follower-version", config["follower_version"],
    ]
    if task == "read_only":
        return [python, "-Xutf8", str(ROOT / "examples" / "nero" / "inspect_effectors.py"), *common]
    if task == "disable":
        return [
            python,
            "-Xutf8",
            str(ROOT / "examples" / "nero" / "disable_pair.py"),
            "--both-supported-empty-confirmed",
            *common,
        ]
    if task == "teleoperate":
        command = [
            python,
            "-Xutf8",
            str(ROOT / "examples" / "nero" / "teleoperate.py"),
            "--interface", "agx_cando",
            "--leader-channel", config["leader_channel"],
            "--follower-channel", config["follower_channel"],
            "--leader-firmware", config["leader_firmware"],
            "--follower-firmware", config["follower_firmware"],
            "--mapping", config["mapping"],
            "--fps", str(config["fps"]),
            "--speed-percent", str(config["speed_percent"]),
            "--gripper-force", str(config["gripper_force"]),
            "--max-joint-speed", str(config["max_joint_speed"]),
        ]
        if config["return_to_start"]:
            command.append("--return-to-start")
        if config["duration"] is not None:
            command.extend(["--duration", f"{config['duration']:g}"])
        return command
    raise ConfigError("未知操作")


def _command_label(command: list[str]) -> str:
    display = command[:]
    display[0] = Path(display[0]).name
    return shlex.join(display)


class ControlCenter:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.logs: deque[dict[str, Any]] = deque(maxlen=LOG_LIMIT)
        self.log_id = 0
        self.process: subprocess.Popen[str] | None = None
        self.task: str | None = None
        self.command: list[str] | None = None
        self.started_at: float | None = None
        self.exit_code: int | None = None
        self.stopping = False
        self.verified_signature: str | None = None
        self.last_teleop_config: dict[str, Any] | None = None
        self.environment_ready = importlib.util.find_spec("pyAgxArm") is not None

    def add_log(self, level: str, message: str) -> None:
        with self.lock:
            self.log_id += 1
            self.logs.append(
                {
                    "id": self.log_id,
                    "time": time.strftime("%H:%M:%S"),
                    "level": level,
                    "message": message.rstrip(),
                }
            )

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            running = self.process is not None and self.process.poll() is None
            return {
                "running": running,
                "busy": running or self.stopping,
                "task": self.task,
                "status": (
                    "stopping" if self.stopping
                    else self.task if running
                    else "idle" if self.exit_code is None
                    else "finished" if self.exit_code == 0
                    else "failed"
                ),
                "pid": self.process.pid if running else None,
                "started_at": self.started_at,
                "exit_code": self.exit_code,
                "command": _command_label(self.command) if self.command else None,
                "verified_signature": self.verified_signature,
                "environment_ready": self.environment_ready,
            }

    def logs_after(self, after: int) -> list[dict[str, Any]]:
        with self.lock:
            return [entry for entry in self.logs if entry["id"] > after]

    def start(self, task: str, config: dict[str, Any]) -> dict[str, Any]:
        command = build_command(task, config)
        with self.lock:
            if self.stopping:
                raise ConfigError("正在停止上一项操作，请稍候")
            if self.process is not None and self.process.poll() is None:
                raise ConfigError("已有硬件操作正在运行")
            if task == "teleoperate":
                signature = config_signature(config)
                if self.verified_signature != signature:
                    raise ConfigError("当前通道和固件组合尚未通过只读身份检查")
                self.last_teleop_config = dict(config)

            self.add_log("control", f"启动 {task}: {_command_label(command)}")
            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            env["PYTHONUNBUFFERED"] = "1"
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            self.process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
            )
            self.task = task
            self.command = command
            self.started_at = time.time()
            self.exit_code = None
            process = self.process
            threading.Thread(target=self._read_output, args=(process,), daemon=True).start()
            threading.Thread(target=self._wait, args=(process, task, config), daemon=True).start()
            return self.snapshot()

    def _read_output(self, process: subprocess.Popen[str]) -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            self.add_log("process", line)

    def _wait(self, process: subprocess.Popen[str], task: str, config: dict[str, Any]) -> None:
        code = process.wait()
        with self.lock:
            if self.process is process:
                self.process = None
                self.task = None
                self.exit_code = code
                self.started_at = None
            if task == "read_only" and code == 0:
                self.verified_signature = config_signature(config)
                self.add_log("control", "只读身份检查通过；该通道/固件组合可用于启动遥操作")
            elif task == "read_only":
                self.verified_signature = None
            if task == "disable":
                if code == 0:
                    self.add_log("control", "失能流程完成，关节和 AgxGripper 已由更新反馈确认")
                else:
                    self.add_log("warning", "失能结果 UNCONFIRMED；保持机械臂支撑并现场核对")
        self.add_log("control", f"进程退出，代码 {code}")

    def _interrupt(self, process: subprocess.Popen[str]) -> None:
        self.add_log("control", "发送中断，等待现有程序执行自身清理")
        try:
            if os.name == "nt":
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                process.send_signal(signal.SIGINT)
        except Exception:
            process.terminate()
        try:
            process.wait(timeout=6)
        except subprocess.TimeoutExpired:
            self.add_log("warning", "程序未在 6 秒内退出，强制终止；仍需显式确认失能")
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()

    def stop(self, disable_after: bool) -> dict[str, Any]:
        with self.lock:
            process = self.process
            task = self.task
            config = self.last_teleop_config if task == "teleoperate" else None
            if process is None or process.poll() is not None:
                raise ConfigError("当前没有正在运行的硬件操作")
            self.stopping = True
        threading.Thread(
            target=self._stop_worker,
            args=(process, task, config, disable_after),
            daemon=True,
        ).start()
        return {"accepted": True, "task": task, "disable_after": disable_after}

    def _stop_worker(
        self,
        process: subprocess.Popen[str],
        task: str | None,
        config: dict[str, Any] | None,
        disable_after: bool,
    ) -> None:
        try:
            self._interrupt(process)
            process.wait(timeout=2)
        except Exception:
            self.add_log("warning", "停止过程中读取进程状态失败")
        finally:
            with self.lock:
                if self.process is process:
                    self.process = None
                    self.task = None
                    self.exit_code = process.returncode
                    self.started_at = None
                self.stopping = False
        if disable_after:
            if task == "teleoperate" and config is not None:
                try:
                    self.start("disable", config)
                except Exception as exc:
                    self.add_log("warning", f"无法启动显式失能流程: {exc}")
            else:
                self.add_log("warning", "当前任务不是遥操作；请使用“请求失能”并选择设备配置")

    def request_disable(self, config: dict[str, Any]) -> dict[str, Any]:
        normalized = normalize_config({**config, "safety_confirmed": True}, "disable")
        return self.start("disable", normalized)

    def shutdown_cleanup(self) -> None:
        with self.lock:
            process = self.process
            task = self.task
            config = self.last_teleop_config if task == "teleoperate" else None
        if process is not None and process.poll() is None:
            self._interrupt(process)
        if task == "teleoperate" and config is not None:
            self.add_log("warning", "控制台退出；正在请求关节和 AgxGripper 显式失能")
            command = build_command("disable", config)
            try:
                result = subprocess.run(command, cwd=ROOT, timeout=20, check=False)
                if result.returncode == 0:
                    self.add_log("control", "控制台退出清理已确认失能")
                else:
                    self.add_log("warning", "控制台退出清理未能确认 AgxGripper 与关节失能，必须现场核对")
            except Exception as exc:
                self.add_log("warning", f"控制台退出清理失败: {exc}")


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "NeroControlCenter/1.0"
    controller: ControlCenter

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _is_local(self) -> bool:
        host = self.headers.get("Host", "").split(":")[0].lower()
        origin = self.headers.get("Origin")
        if host not in {"127.0.0.1", "localhost", "::1"}:
            return False
        if origin and urlparse(origin).hostname not in {"127.0.0.1", "localhost", "::1"}:
            return False
        return True

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:
        if not self._is_local():
            self._json(403, {"error": "只允许从本机访问"})
            return
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send(200, UI_FILE.read_bytes(), "text/html; charset=utf-8")
        elif parsed.path == "/static/control_center.css":
            path = Path(__file__).with_name("control_center_ui") / "control_center.css"
            self._send(200, path.read_bytes(), "text/css; charset=utf-8")
        elif parsed.path == "/static/control_center.js":
            path = Path(__file__).with_name("control_center_ui") / "control_center.js"
            self._send(200, path.read_bytes(), "text/javascript; charset=utf-8")
        elif parsed.path == "/assets/lerobot-logo.png":
            path = ROOT / "media" / "lerobot-logo-thumbnail.png"
            self._send(200, path.read_bytes(), "image/png")
        elif parsed.path == "/api/state":
            self._json(200, self.controller.snapshot())
        elif parsed.path == "/api/logs":
            after = int(parse_qs(parsed.query).get("after", ["0"])[0])
            self._json(200, {"logs": self.controller.logs_after(after)})
        else:
            self._json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if not self._is_local():
            self._json(403, {"error": "只允许从本机访问"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 65536:
                raise ConfigError("请求内容过大")
            payload = json.loads(self.rfile.read(length) or b"{}")
            action = payload.get("action")
            if action in {"read_only", "teleoperate", "disable"}:
                config = normalize_config(payload.get("config", {}), action)
                result = self.controller.start(action, config)
            elif action in {"stop", "stop_and_disable"}:
                if action == "stop_and_disable" and payload.get("safety_confirmed") is not True:
                    raise ConfigError("停止并失能前必须逐项确认实体安全条件")
                result = self.controller.stop(disable_after=action == "stop_and_disable")
            else:
                raise ConfigError("未知操作")
            self._json(200, result)
        except ConfigError as exc:
            self._json(400, {"error": str(exc)})
        except json.JSONDecodeError:
            self._json(400, {"error": "JSON 格式错误"})
        except Exception as exc:
            self.controller.add_log("warning", f"HTTP 操作失败: {exc}")
            self._json(500, {"error": str(exc)})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local-only NERO control center")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    controller = ControlCenter()
    controller.add_log("control", "NERO 本机控制台已就绪。网页不是实体急停。")
    if not controller.environment_ready:
        controller.add_log("warning", "当前 Python 未找到 pyAgxArm；请从 lerobot conda 环境启动控制台")

    handler = type("BoundRequestHandler", (RequestHandler,), {"controller": controller})
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"NERO control center: {url}", flush=True)
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        print("\nStopping control center...", flush=True)
    finally:
        server.shutdown()
        server.server_close()
        controller.shutdown_cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




