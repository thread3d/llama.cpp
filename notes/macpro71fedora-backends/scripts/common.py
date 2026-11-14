"""Shared definitions for the MacPro71Fedora backend study.

One place defines the machine's device topology so the correctness sweep, the
performance sweep and the capacity probe cannot drift apart.

The Mac Pro 7,1 on this machine has three AMD GPUs:

    2x Radeon Pro W6800X in one MPX "Duo" module   gfx1030, wave32, 60 CU, 32 GiB each
    1x Radeon Pro Vega II                          gfx906,  wave64,          32 GiB

The build targets gfx1030 only, so the Vega II must be kept off every device
list (the ROCm build segfaults at model load if it is left visible). The two
backends number the cards differently, which is why the selectors below are not
interchangeable:

    ROCm   0,1 = the Duo,      2 = Vega II      -> HIP_VISIBLE_DEVICES=0 / 1 / 0,1
    Vulkan 0   = Vega II,      1,2 = the Duo    -> GGML_VK_VISIBLE_DEVICES=1 / 2 / 1,2
"""

import os
import re

REPO = "/home/threaded/projects/Llama.cpp/llama.cpp"
PARENT = "/home/threaded/projects/Llama.cpp"
MODELS = PARENT + "/.models"
WORK = PARENT + "/logs/backend-matrix"

# topology -> how to launch it.
#   build: which CMake build directory supplies the binaries
#   env:   the device filter for that backend
#   args:  extra command line the runner appends
TOPOLOGIES = {
    "cpu": dict(
        build="vulkan",
        env={},
        args=["-dev", "none"],
        label="CPU (28 cores / 56 threads)",
        color="#444444",
        marker="o",
    ),
    "rocm-die1": dict(
        build="rocm",
        env={"HIP_VISIBLE_DEVICES": "0"},
        args=["-ngl", "99"],
        label="ROCm, Duo die 1",
        color="#1f5fa8",
        marker="s",
    ),
    "rocm-die2": dict(
        build="rocm",
        env={"HIP_VISIBLE_DEVICES": "1"},
        args=["-ngl", "99"],
        label="ROCm, Duo die 2",
        color="#4a90d9",
        marker="s",
    ),
    "rocm-both": dict(
        build="rocm",
        env={"HIP_VISIBLE_DEVICES": "0,1"},
        args=["-ngl", "99"],
        label="ROCm, both dies (layer split)",
        color="#12395f",
        marker="s",
    ),
    "vk-die1": dict(
        build="vulkan",
        env={"GGML_VK_VISIBLE_DEVICES": "1"},
        args=["-ngl", "99"],
        label="Vulkan, Duo die 1",
        color="#1b7f4b",
        marker="^",
    ),
    "vk-die2": dict(
        build="vulkan",
        env={"GGML_VK_VISIBLE_DEVICES": "2"},
        args=["-ngl", "99"],
        label="Vulkan, Duo die 2",
        color="#4aa96c",
        marker="^",
    ),
    "vk-both": dict(
        build="vulkan",
        env={"GGML_VK_VISIBLE_DEVICES": "1,2"},
        args=["-ngl", "99"],
        label="Vulkan, both dies (layer split)",
        color="#0f4d2e",
        marker="^",
    ),
}

# order used by every chart and table
ORDER = ["cpu", "rocm-die1", "rocm-die2", "rocm-both", "vk-die1", "vk-die2", "vk-both"]

ANSI = re.compile(r"\x1b\[[0-9;]*m")


def binary(topology, tool):
    """Absolute path of `tool` in the build that serves this topology."""
    return os.path.join(REPO, "build-" + TOPOLOGIES[topology]["build"], "bin", tool)


def launch(topology, tool, extra=()):
    """(argv, env) that runs `tool` on this topology.

    The whole environment is inherited and only the device filter is added, so a
    caller can still override anything it needs.
    """
    spec = TOPOLOGIES[topology]
    env = dict(os.environ)
    env.update(spec["env"])
    argv = [binary(topology, tool)] + list(spec["args"]) + list(extra)
    return argv, env


def log_devices(topology, tool):
    """The device line llama.cpp prints for this topology, for the record."""
    import subprocess

    argv, env = launch(topology, tool, ["--list-devices"])
    out = subprocess.run(argv, env=env, capture_output=True, text=True)
    return ANSI.sub("", out.stdout + out.stderr).strip()
