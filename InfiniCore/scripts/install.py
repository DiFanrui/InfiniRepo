import os
import subprocess
import platform
import sys
from set_env import (
    set_env,
    set_env_by_config,
)

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_DIR)

def run_cmd(cmd):
    env = {**os.environ, "XMAKE_ROOT": "y"}
    subprocess.run(cmd, text=True, encoding="utf-8", check=True, shell=True, env=env)


def install(xmake_config_flags=""):
    set_env_by_config(xmake_config_flags)
    run_cmd(f"xmake f -y {xmake_config_flags} -cv")
    run_cmd(f"xmake -y -j4")
    run_cmd(f"xmake install -y")
    run_cmd(f"xmake build -y -j4 infiniop-test")
    run_cmd(f"xmake install -y infiniop-test")


if __name__ == "__main__":
    set_env()
    install(" ".join(sys.argv[1:]))
