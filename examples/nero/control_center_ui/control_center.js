"use strict";

const elements = {
  leaderChannel: document.querySelector("#leaderChannel"),
  followerChannel: document.querySelector("#followerChannel"),
  leaderFirmware: document.querySelector("#leaderFirmware"),
  followerFirmware: document.querySelector("#followerFirmware"),
  fps: document.querySelector("#fps"),
  speedPercent: document.querySelector("#speedPercent"),
  maxJointSpeed: document.querySelector("#maxJointSpeed"),
  gripperForce: document.querySelector("#gripperForce"),
  returnToStart: document.querySelector("#returnToStart"),
  duration: document.querySelector("#duration"),
  readOnlyBtn: document.querySelector("#readOnlyBtn"),
  teleopBtn: document.querySelector("#teleopBtn"),
  disableBtn: document.querySelector("#disableBtn"),
  stopBtn: document.querySelector("#stopBtn"),
  stopDisableBtn: document.querySelector("#stopDisableBtn"),
  clearLogsBtn: document.querySelector("#clearLogsBtn"),
  checks: [
    document.querySelector("#checkSupport"),
    document.querySelector("#checkGrippers"),
    document.querySelector("#checkArea"),
  ],
  envBadge: document.querySelector("#envBadge"),
  runBadge: document.querySelector("#runBadge"),
  identityBadge: document.querySelector("#identityBadge"),
  statusText: document.querySelector("#statusText"),
  taskText: document.querySelector("#taskText"),
  pidText: document.querySelector("#pidText"),
  exitText: document.querySelector("#exitText"),
  jointList: document.querySelector("#jointList"),
  gripperValue: document.querySelector("#gripperValue"),
  logView: document.querySelector("#logView"),
  toast: document.querySelector("#toast"),
};

let mapping = "relative";
let latestLogId = 0;
let state = null;
let toastTimer = null;
const visibleLogs = new Map();

for (let index = 1; index <= 7; index += 1) {
  const row = document.createElement("div");
  row.className = "joint-row";
  row.innerHTML = `
    <span class="joint-name">J${index}</span>
    <div class="joint-track"><span class="joint-marker"></span></div>
    <span class="joint-value">--</span>`;
  elements.jointList.appendChild(row);
}

document.querySelectorAll("[data-mapping]").forEach((button) => {
  button.addEventListener("click", () => {
    mapping = button.dataset.mapping;
    document.querySelectorAll("[data-mapping]").forEach((item) => {
      item.classList.toggle("active", item === button);
    });
  });
});

elements.fps.addEventListener("input", () => {
  document.querySelector("#fpsValue").textContent = `${elements.fps.value} Hz`;
});
elements.speedPercent.addEventListener("input", () => {
  document.querySelector("#speedValue").textContent = `${elements.speedPercent.value}%`;
});
elements.maxJointSpeed.addEventListener("input", () => {
  document.querySelector("#jointSpeedValue").textContent = `${Number(elements.maxJointSpeed.value).toFixed(2)} rad/s`;
});
elements.gripperForce.addEventListener("input", () => {
  document.querySelector("#forceValue").textContent = `${Number(elements.gripperForce.value).toFixed(1)} N`;
});
elements.clearLogsBtn.addEventListener("click", () => {
  visibleLogs.clear();
  elements.logView.replaceChildren();
});

function showToast(message, error = false) {
  clearTimeout(toastTimer);
  elements.toast.textContent = message;
  elements.toast.classList.toggle("error", error);
  elements.toast.classList.add("show");
  toastTimer = setTimeout(() => elements.toast.classList.remove("show"), 4200);
}

function currentConfig() {
  const config = {
    leader_channel: elements.leaderChannel.value,
    follower_channel: elements.followerChannel.value,
    leader_firmware: elements.leaderFirmware.value,
    follower_firmware: elements.followerFirmware.value,
  };
  if (!state?.running) {
    config.fps = Number(elements.fps.value);
    config.speed_percent = Number(elements.speedPercent.value);
    config.max_joint_speed = Number(elements.maxJointSpeed.value);
    config.gripper_force = Number(elements.gripperForce.value);
    config.mapping = mapping;
    config.return_to_start = elements.returnToStart.checked;
    config.duration = elements.duration.value === "" ? null : Number(elements.duration.value);
  }
  return config;
}

function configSignature(config) {
  return [
    config.leader_channel,
    config.follower_channel,
    config.leader_firmware,
    config.follower_firmware,
  ].join("|");
}

function verifiedSignature() {
  const value = state?.verified_signature;
  return value ? value.split("|").slice(0, 4).join("|") : null;
}

function safetyConfirmed() {
  return elements.checks.every((checkbox) => checkbox.checked);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {"Content-Type": "application/json", ...(options.headers || {})},
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

async function perform(action) {
  try {
    const body = {
      action,
      safety_confirmed: safetyConfirmed(),
      config: currentConfig(),
    };
    await api("/api/action", {
      method: "POST",
      body: JSON.stringify(body),
    });
    showToast(
      action === "teleoperate"
        ? "遥操作已启动"
        : action === "read_only"
          ? "只读身份检查已启动"
          : action === "stop"
            ? "已请求停止"
            : "已请求停止并显式失能"
    );
    await refresh();
  } catch (error) {
    showToast(error.message, true);
  }
}

elements.readOnlyBtn.addEventListener("click", () => perform("read_only"));
elements.teleopBtn.addEventListener("click", () => perform("teleoperate"));
elements.disableBtn.addEventListener("click", () => perform("disable"));
elements.stopBtn.addEventListener("click", () => perform("stop"));
elements.stopDisableBtn.addEventListener("click", () => {
  perform(state?.running ? "stop_and_disable" : "disable");
});

function setBadge(element, text, kind) {
  element.textContent = text;
  element.className = `badge ${kind}`;
}

function taskLabel(task) {
  return {
    read_only: "只读身份检查",
    teleoperate: "双臂遥操作",
    disable: "关节与末端失能",
  }[task] || "无";
}

function updateControls() {
  const running = Boolean(state?.running);
  const busy = Boolean(state?.busy);
  const identityOk = verifiedSignature() === configSignature(currentConfig());
  const ready = safetyConfirmed() && identityOk && !busy;

  elements.readOnlyBtn.disabled = busy;
  elements.teleopBtn.disabled = !ready;
  elements.disableBtn.disabled = busy || !safetyConfirmed();
  elements.stopBtn.disabled = !running;
  elements.stopDisableBtn.disabled = busy || (!running && !safetyConfirmed());

  setBadge(
    elements.envBadge,
    state?.environment_ready ? "pyAgxArm 就绪" : "硬件环境缺失",
    state?.environment_ready ? "good" : "danger"
  );
  setBadge(
    elements.identityBadge,
    identityOk ? "身份已确认" : "身份未确认",
    identityOk ? "good" : "warning"
  );

  const statusText = {
    idle: "空闲",
    read_only: "只读检查运行中",
    teleoperate: "遥操作运行中",
    disable: "失能确认运行中",
    stopping: "正在停止",
    finished: "上一项已完成",
    failed: "上一项失败",
  }[state?.status] || "空闲";
  setBadge(
    elements.runBadge,
    statusText,
    state?.status === "failed" ? "danger" : running ? "good" : state?.status === "stopping" ? "warning" : "neutral"
  );
  elements.statusText.textContent = statusText;
  elements.taskText.textContent = taskLabel(state?.task);
  elements.pidText.textContent = state?.pid || "-";
  elements.exitText.textContent = state?.exit_code ?? "-";
}

function parseJoints(message) {
  const match = message.match(/FOLLOWER feedback \| joints\(deg\): (.*?) \| gripper=(.*)$/);
  if (!match) return;
  const values = [...match[1].matchAll(/J\d+=(-?\d+(?:\.\d+)?)/g)].map((item) => Number(item[1]));
  if (values.length !== 7) return;
  document.querySelectorAll(".joint-row").forEach((row, index) => {
    const value = values[index];
    row.querySelector(".joint-value").textContent = `${value.toFixed(2)}°`;
    row.querySelector(".joint-marker").style.left = `${Math.max(0, Math.min(100, (value + 180) / 360 * 100))}%`;
  });
  elements.gripperValue.textContent = `夹爪 ${match[2]}`;
}

function renderLog(entry) {
  if (visibleLogs.has(entry.id)) return;
  visibleLogs.set(entry.id, entry);
  const line = document.createElement("div");
  line.className = `log-line ${entry.level}`;
  const time = document.createElement("span");
  time.className = "time";
  time.textContent = entry.time;
  const level = document.createElement("span");
  level.className = "level";
  level.textContent = entry.level.toUpperCase();
  const message = document.createElement("span");
  message.className = "message";
  message.textContent = entry.message;
  line.append(time, level, message);
  elements.logView.appendChild(line);
  parseJoints(entry.message);
  while (elements.logView.childElementCount > 3000) {
    elements.logView.firstElementChild.remove();
  }
  elements.logView.scrollTop = elements.logView.scrollHeight;
}

async function refreshLogs() {
  try {
    const payload = await api(`/api/logs?after=${latestLogId}`);
    payload.logs.forEach((entry) => {
      latestLogId = Math.max(latestLogId, entry.id);
      renderLog(entry);
    });
  } catch (error) {
    showToast(error.message, true);
  }
}

async function refresh() {
  try {
    state = await api("/api/state");
    updateControls();
    await refreshLogs();
  } catch (error) {
    showToast(error.message, true);
  }
}

elements.checks.forEach((checkbox) => checkbox.addEventListener("change", updateControls));
[
  elements.leaderChannel,
  elements.followerChannel,
  elements.leaderFirmware,
  elements.followerFirmware,
].forEach((input) => input.addEventListener("change", updateControls));

refresh();
setInterval(refresh, 800);
