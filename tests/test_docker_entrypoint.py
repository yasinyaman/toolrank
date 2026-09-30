"""The toolrank-vllm image's entrypoint, in dry-run mode: which vLLM and toolrank commands it runs."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ENTRYPOINT = Path(__file__).resolve().parents[1] / "deploy" / "docker" / "entrypoint-vllm.sh"
pytestmark = pytest.mark.skipif(
    not ENTRYPOINT.exists() or shutil.which("bash") is None, reason="needs the repo's deploy/ and bash"
)


SETTINGS = ("TOOLRANK_FP8", "TOOLRANK_EMB_MODEL", "VLLM_EXTRA_ARGS", "VLLM_GPU_MEMORY_UTILIZATION")


def _dry_run(env, *argv, path=None):
    inherited = {k: v for k, v in os.environ.items() if k not in SETTINGS}  # only the test's settings
    env = {**inherited, "TOOLRANK_ENTRYPOINT_DRY_RUN": "1", **env}
    if path is not None:
        env["PATH"] = f"{path}{os.pathsep}{env['PATH']}"
    out = subprocess.run(
        ["bash", str(ENTRYPOINT), *argv], env=env, capture_output=True, text=True, check=True
    )
    vllm, toolrank = out.stdout.strip().splitlines()
    return vllm.split(), toolrank


def test_fp8_and_bf16_are_served_under_different_names():
    vllm, toolrank = _dry_run({"TOOLRANK_FP8": "1"}, "serve", "--data", "/data")
    assert vllm[:3] == ["vllm", "serve", "Qwen/Qwen3-Embedding-8B"] and "--quantization" in vllm
    assert vllm[vllm.index("--served-model-name") + 1] == "qwen3-emb-fp8"
    assert vllm[vllm.index("--host") + 1] == "127.0.0.1"  # vLLM stays inside the container
    assert "TOOLRANK_EMB_MODEL=qwen3-emb-fp8" in toolrank and toolrank.endswith("toolrank serve --data /data")
    vllm, toolrank = _dry_run({"TOOLRANK_FP8": "0"}, "serve", "--data", "/data")
    assert "--quantization" not in vllm and vllm[vllm.index("--served-model-name") + 1] == "qwen3-emb"
    assert "TOOLRANK_EMB_MODEL=qwen3-emb " in toolrank  # a separate cache namespace from the FP8 vectors


def test_extra_vllm_flags_and_memory_share_pass_through():
    env = {"VLLM_EXTRA_ARGS": "--enforce-eager --seed 1", "VLLM_GPU_MEMORY_UTILIZATION": "0.2"}
    vllm, _ = _dry_run(env, "ingest", "mcp", "--config", "/config/toolrank.json", "--out", "/data")
    assert vllm[-3:] == ["--enforce-eager", "--seed", "1"] and "0.2" in vllm


AS_TOOLRANK = ENTRYPOINT.with_name("as-toolrank.sh")
UV = "env -u UV_CACHE_DIR -u UV_OVERRIDE -u UV_INDEX_STRATEGY"  # root's cache, vLLM's override
DROP = "--clear-groups --no-new-privs"
ROOTFUL, ROOTLESS = "0 0 4294967295", "0 1000 1"  # /proc/self/uid_map
# the commands the script asks about the container, answering from the test; the ones that would
# change it (install, find, setpriv) record their arguments in $REC instead
STUBS = {
    "id": 'case "$*" in -u) echo $AS_UID ;; "-u toolrank" | "-g toolrank") echo 2001 ;; esac',
    "stat": 'for a; do last=$a; done; echo "$last" >> "$REC/stat"; [ -n "$AS_OWNER" ] && echo "$AS_OWNER"',
    "cat": 'case "$1" in /proc/self/uid_map) echo "$AS_UID_MAP" ;; *) exec /bin/cat "$@" ;; esac',
    "install": 'echo "$@" >> "$REC/install"; exit ${AS_INSTALL_EXIT:-0}',
    "find": 'echo "$@" >> "$REC/find"; exit ${AS_FIND_EXIT:-0}',
    "setpriv": 'echo "$@" >> "$REC/setpriv"; while [ "${1#--}" != "$1" ]; do shift; done; exec "$@"',
}


class _Box:
    """as-toolrank.sh run with the stubs first on the PATH; ``rec(name)`` is what a stub was given."""

    def __init__(self, tmp_path):
        self.bin, self.recs = tmp_path / "bin", tmp_path / "rec"
        self.bin.mkdir()
        self.recs.mkdir()
        for name, body in STUBS.items():
            (self.bin / name).write_text(f"#!/bin/sh\n{body}\n")
            (self.bin / name).chmod(0o755)

    def run(self, *argv, uid=0, owner="0:0 /data", uid_map=ROOTFUL, home=None, dry=True, **exits):
        env = {k: v for k, v in os.environ.items() if k != "TOOLRANK_ENTRYPOINT_DRY_RUN"}
        env.update(
            PATH=f"{self.bin}{os.pathsep}{env['PATH']}", REC=str(self.recs), AS_UID=str(uid),
            AS_OWNER=owner or "", AS_UID_MAP=uid_map, **{f"AS_{k.upper()}_EXIT": str(v) for k, v in exits.items()},
        )  # fmt: skip
        if dry:
            env["TOOLRANK_ENTRYPOINT_DRY_RUN"] = "1"
        if home is not None:
            env["HOME"] = home
        for f in self.recs.iterdir():
            f.unlink()
        return subprocess.run(
            ["bash", str(AS_TOOLRANK), *(argv or ("toolrank", "serve"))],
            env=env,
            capture_output=True,
            text=True,
        )

    def line(self, *argv, **kw):
        out = self.run(*argv, **kw)
        assert out.returncode == 0, out.stderr
        return out.stdout.strip()

    def rec(self, name):
        f = self.recs / name
        return f.read_text().splitlines() if f.exists() else []


def test_the_entrypoint_leaves_the_user_to_the_toolrank_wrapper():
    """vLLM runs as the container's user; `toolrank` on the image's PATH is as-toolrank's wrapper."""
    vllm, toolrank = _dry_run({}, "serve", "--data", "/data")
    assert "setpriv" not in vllm and toolrank.endswith(" toolrank serve --data /data")


def test_a_root_container_never_runs_toolrank_as_root(tmp_path):
    box = _Box(tmp_path)
    image_user = f"{UV} HOME=/home/toolrank setpriv --reuid=2001 --regid=2001 {DROP} toolrank serve"
    # root's directory (a bind mount Docker created), the image's own, or none at all: the image's user
    for owner in ("0:0 /data", "2001:2001 /data", None):
        assert box.line(owner=owner) == image_user
    # someone's directory, bind-mounted: toolrank runs as its owner, with a home only root can replace
    theirs = box.line(owner="1000:984 /data")
    assert theirs == f"{UV} HOME=/home/toolrank-1000 setpriv --reuid=1000 --regid=984 {DROP} toolrank serve"
    # `sudo mkdir` + `chown $USER`: theirs, but root's group, which toolrank never joins
    assert "--reuid=1000 --regid=2001 " in box.line(owner="1000:0 /data")
    # a directory of a user this container cannot map (under a rootless engine, host root's): the image's user
    out = box.run(owner="65534:65534 /data")
    assert out.stdout.strip() == image_user and "cannot map" in out.stderr


def test_the_user_follows_the_directory_the_command_writes_to(tmp_path):
    box = _Box(tmp_path)
    tools, out = tmp_path / "tools", tmp_path / "results" / "new" / "run.json"
    tools.mkdir()
    asked = lambda *argv: (box.line("toolrank", *argv), box.rec("stat")[-1])[1]  # noqa: E731
    assert asked("serve", "--data", str(tools)) == str(tools)  # serve and search write where they read
    assert asked("search", f"--data={tools}", "q") == str(tools)
    assert asked("eval", "--data", str(tools)) != str(tools)  # eval only reads --data: the image's /data
    assert asked("eval", "--data", str(tools), "--out", str(out)) == str(tmp_path)  # nearest parent there is
    assert asked("ingest", "mcp", f"--out={tools}") == str(tools)


def test_under_a_rootless_engine_roots_directory_is_the_users_own(tmp_path):
    box = _Box(tmp_path)
    # uid 0 is the user who runs the engine: any other uid would lock them out of their directory
    assert box.line(owner="0:0 /data", uid_map=ROOTLESS) == f"{UV} toolrank serve"
    out = box.run("echo", "ran", owner="0:0 /data", uid_map=ROOTLESS, dry=False)
    assert out.stdout.strip() == "ran" and not box.rec("find") and not box.rec("setpriv")
    assert "--reuid=2001" in box.line(owner="2001:2001 /data", uid_map=ROOTLESS)  # a named volume still drops


def test_only_roots_files_on_the_data_volume_change_hands_and_failures_are_warnings(tmp_path):
    box = _Box(tmp_path)
    ok = box.run("echo", "ran", dry=False)
    assert (ok.returncode, ok.stdout.strip(), ok.stderr) == (0, "ran", "")
    assert box.rec("find") == ["/data -xdev -user root -exec chown -h 2001:2001 {} +"]
    assert box.rec("install") == [
        "-d -m 0750 -o 2001 -g 2001 /home/toolrank"
    ]  # only because it is missing here
    assert box.rec("setpriv") == [f"--reuid=2001 --regid=2001 {DROP} echo ran"]
    # a symlink that does not resolve, a read-only mount, a read-only root filesystem: said, not fatal
    failing = box.run("echo", "ran", dry=False, find=1, install=1)
    assert (failing.returncode, failing.stdout.strip()) == (0, "ran")
    assert "could not be handed to uid 2001" in failing.stderr and "no home for uid 2001" in failing.stderr
    # another mount, or the image's own filesystem: nothing is re-owned, whoever owns it
    for owner in ("0:0 /tools", "1000:1000 /tools", "0:0 /"):
        assert box.run("echo", "ran", owner=owner, dry=False).stdout.strip() == "ran" and not box.rec("find")
    theirs = box.run("echo", "ran", owner="1000:1000 /data", dry=False)
    assert (
        box.rec("find") == ["/data -xdev -user root -exec chown -h 1000:1000 {} +"] and theirs.returncode == 0
    )


def test_a_container_started_as_another_user_keeps_it_and_gets_a_home(tmp_path):
    box = _Box(tmp_path)
    assert box.line(uid=1000, home=str(tmp_path)) == f"{UV} toolrank serve"
    if not os.access("/", os.W_OK):  # no passwd entry: HOME is /, which it cannot write (root can)
        assert "toolrank-home.XXXXXX toolrank serve" in box.line(uid=1234, home="/")
        made = box.run("sh", "-c", 'echo "$HOME"', uid=1234, home="/", dry=False).stdout.strip()
        assert (
            "toolrank-home." in made and Path(made).is_dir() and oct(Path(made).stat().st_mode)[-3:] == "700"
        )
        Path(made).rmdir()
    assert not box.rec("setpriv") and not box.rec("find")
