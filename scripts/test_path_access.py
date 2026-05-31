import os, sys
rank = int(os.environ.get("RANK", 0))
if rank != 0:
    sys.exit(0)

base = "/playpen/huggingface/hub"
p27 = f"{base}/models--Qwen--Qwen3.5-27B/snapshots/fc05daec18b0a78c049392ed2e771dde82bdf654"
p9  = f"{base}/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"

print(f"base isdir:  {os.path.isdir(base)}")
print(f"27B isdir:   {os.path.isdir(p27)}")
print(f"9B  isdir:   {os.path.isdir(p9)}")

try:
    entries = os.listdir(base)
    qwen = [e for e in entries if "Qwen" in e]
    print(f"Qwen entries in hub: {qwen}")
except Exception as e:
    print(f"listdir failed: {e}")

# Try to stat the 27B snapshot directly
import subprocess
result = subprocess.run(["ls", "-la", os.path.dirname(p27)], capture_output=True, text=True)
print("ls snapshots:", result.stdout.strip() or result.stderr.strip())
