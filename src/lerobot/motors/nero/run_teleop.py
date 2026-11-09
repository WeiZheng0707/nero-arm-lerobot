import sys
import time
import traceback

print(f"Python: {sys.version}", flush=True)
print(f"平台: {sys.platform}", flush=True)

try:
    from pyAgxArm import AgxArmFactory, create_agx_arm_config
except ImportError as e:
    print(f"[FATAL] 无法导入 pyAgxArm: {e}", flush=True)
    sys.exit(1)

# 配置参数（Windows 默认 agx_cando）
INTERFACE = "agx_cando"
CHANNEL = "0"
BITRATE = 1_000_000

print(f"使用配置: interface={INTERFACE}, channel={CHANNEL}, bitrate={BITRATE}", flush=True)

cfg = create_agx_arm_config(
    robot="nero",
    comm="can",
    interface=INTERFACE,
    channel=CHANNEL,
    bitrate=BITRATE,
)

print("创建 arm 实例...", flush=True)
arm = AgxArmFactory.create_arm(cfg)

print("尝试连接...", flush=True)
try:
    arm.connect()
    print("✅ 连接成功！", flush=True)
except Exception as e:
    print(f"❌ 连接失败: {type(e).__name__}: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

# 连接成功后，尝试打开 CAN 推送
print("调用 set_normal_mode() 打开 CAN 推送...", flush=True)
try:
    arm.set_normal_mode()
    print("   set_normal_mode 调用完成", flush=True)
except Exception as e:
    print(f"   set_normal_mode 失败: {type(e).__name__}: {e}", flush=True)

time.sleep(1.0)

# 尝试读取固件
print("尝试读取固件版本...", flush=True)
fw = None
for i in range(10):
    try:
        fw = arm.get_firmware(timeout=0.5)
    except Exception as e:
        print(f"   get_firmware 异常: {type(e).__name__}: {e}", flush=True)
        fw = None
    if fw is not None:
        break
    time.sleep(0.3)
if fw is not None:
    print(f"   固件信息: {fw}", flush=True)
else:
    print("   未读到固件信息（可能推送未开或超时）", flush=True)

# 尝试读取关节角
print("尝试读取关节角...", flush=True)
ja = None
for i in range(10):
    try:
        ja = arm.get_joint_angles()
    except Exception as e:
        print(f"   get_joint_angles 异常: {type(e).__name__}: {e}", flush=True)
        ja = None
    if ja is not None:
        break
    time.sleep(0.3)
if ja is not None:
    print(f"   关节角(rad): {ja.msg}", flush=True)
    print(f"   数据频率: {ja.hz:.1f} Hz", flush=True)
else:
    print("   未读到关节角", flush=True)

# 断开连接
print("断开连接...", flush=True)
try:
    arm.disconnect()
    print("✅ 已断开", flush=True)
except Exception as e:
    print(f"   断开时异常: {type(e).__name__}: {e}", flush=True)

print("测试结束。", flush=True)