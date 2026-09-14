"""Stop only a verified process tree belonging to this experiment."""
import sys
import psutil

pid = int(sys.argv[1])
try:
    parent = psutil.Process(pid)
    cmd = parent.cmdline()
except (psutil.NoSuchProcess, psutil.ZombieProcess):
    raise SystemExit(0)
assert cmd[0] == '/data/envs/glm53-vllm-pp2-mtp/bin/python', cmd[0]
assert '/data/model/GLM-5.3' in cmd and 'serve' in cmd, cmd[:3]
tree = parent.children(recursive=True) + [parent]
for process in tree:
    try:
        process.terminate()
    except psutil.NoSuchProcess:
        pass
_, alive = psutil.wait_procs(tree, timeout=5)
for process in alive:
    try:
        process.kill()
    except psutil.NoSuchProcess:
        pass
print('Stopped verified GLM-5.3 vLLM experiment tree:', pid)
