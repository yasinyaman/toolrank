"""The Helm chart (deploy/helm/toolrank) renders what each embedding mode needs. Runs only where
``helm`` is installed (CI has none); scripts/helm_smoke.py installs it on a real cluster."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

CHART = Path(__file__).resolve().parents[1] / "deploy" / "helm" / "toolrank"
yaml = pytest.importorskip("yaml")
pytestmark = pytest.mark.skipif(shutil.which("helm") is None or not CHART.exists(), reason="needs helm")


def render(*sets: str, fails: bool = False):
    args = ["helm", "template", "t", str(CHART), "--namespace", "ns"]
    for s in sets:
        args += (
            ["--set-json", s]
            if s.startswith("{") or "=" in s and s.split("=", 1)[1][:1] in "{["
            else ["--set", s]
        )
    done = subprocess.run(args, capture_output=True, text=True)
    if fails:
        assert done.returncode != 0
        return done.stderr
    assert done.returncode == 0, done.stderr
    docs = [d for d in yaml.safe_load_all(done.stdout) if d]
    return {(d["kind"], d["metadata"]["name"]): d for d in docs}


def container(objs, name="t-toolrank"):
    return objs[("Deployment", name)]["spec"]["template"]["spec"]["containers"][0]


def env(c):
    return {e["name"]: e.get("value", e.get("valueFrom")) for e in c["env"]}


def test_lint_passes():
    done = subprocess.run(
        ["helm", "lint", str(CHART), "--set", "auth.apiKey=x"], capture_output=True, text=True
    )
    assert done.returncode == 0, done.stdout + done.stderr


def test_the_default_runs_vllm_next_to_toolrank_with_the_fp8_profile():
    objs = render("auth.apiKey=secret")
    c = container(objs)
    assert (
        c["image"]
        == "ghcr.io/yasinyaman/toolrank:" + yaml.safe_load((CHART / "Chart.yaml").read_text())["appVersion"]
    )
    e = env(c)
    assert (
        e["TOOLRANK_EMB_URL"] == "http://t-toolrank-embedding:8000/v1"
        and e["TOOLRANK_EMB_MODEL"] == "qwen3-emb-fp8"
    )
    assert e["TOOLRANK_API_KEY"] == {"secretKeyRef": {"name": "t-toolrank-auth", "key": "TOOLRANK_API_KEY"}}
    assert e["TOOLRANK_ALLOWED_HOSTS"].split(",")[:4] == [
        "t-toolrank",
        "t-toolrank.ns",
        "t-toolrank.ns.svc",
        "t-toolrank.ns.svc.cluster.local",
    ]
    assert c["args"][:7] == [
        "serve",
        "--data",
        "/data",
        "--config",
        "/config/toolrank.json",
        "--host",
        "0.0.0.0",
    ]
    assert "--api-keys" not in c["args"] and c["securityContext"]["allowPrivilegeEscalation"] is False
    pod = objs[("Deployment", "t-toolrank")]["spec"]["template"]["spec"]
    assert pod["securityContext"]["runAsUser"] == 1000 and pod["initContainers"][0]["args"][:2] == [
        "ingest",
        "mcp",
    ]
    assert objs[("Deployment", "t-toolrank")]["spec"]["replicas"] == 1
    vllm = container(objs, "t-toolrank-embedding")
    assert "--quantization=fp8" in vllm["args"] and "--served-model-name=qwen3-emb-fp8" in vllm["args"]
    assert vllm["resources"]["limits"]["nvidia.com/gpu"] == 1
    assert json.loads(objs[("ConfigMap", "t-toolrank-config")]["data"]["toolrank.json"])["mcpServers"]["time"]
    assert objs[("Secret", "t-toolrank-auth")]["stringData"] == {"TOOLRANK_API_KEY": "secret"}
    assert (
        objs[("PersistentVolumeClaim", "t-toolrank-data")]["metadata"]["annotations"][
            "helm.sh/resource-policy"
        ]
        == "keep"
    )


def test_bf16_external_and_bundled():
    bf16 = render("auth.apiKey=x", "embedding.profile=bf16")
    assert "--quantization=fp8" not in container(bf16, "t-toolrank-embedding")["args"]
    assert env(container(bf16))["TOOLRANK_EMB_MODEL"] == "qwen3-emb"

    ext = render(
        "auth.apiKey=x", "embedding.mode=external", "embedding.url=http://emb:8000/v1", "embedding.model=m"
    )
    assert ("Deployment", "t-toolrank-embedding") not in ext and (
        "PersistentVolumeClaim",
        "t-toolrank-models",
    ) not in ext
    assert (
        env(container(ext))["TOOLRANK_EMB_URL"] == "http://emb:8000/v1"
        and env(container(ext))["TOOLRANK_EMB_MODEL"] == "m"
    )

    one = render("auth.apiKey=x", "embedding.mode=bundled", "embedding.bundled.image=reg/toolrank-vllm:1")
    c = container(one)
    assert c["image"] == "reg/toolrank-vllm:1" and ("Deployment", "t-toolrank-embedding") not in one
    assert env(c)["TOOLRANK_FP8"] == "1" and env(c)["TOOLRANK_EMB_URL"] == "http://127.0.0.1:8091/v1"
    assert c["resources"]["limits"]["nvidia.com/gpu"] == 1 and "securityContext" not in c  # vLLM keeps root
    assert {m["mountPath"] for m in c["volumeMounts"]} >= {"/data", "/models", "/dev/shm"}
    k3s = render("auth.apiKey=x", "embedding.runtimeClassName=nvidia")
    assert (
        k3s[("Deployment", "t-toolrank-embedding")]["spec"]["template"]["spec"]["runtimeClassName"]
        == "nvidia"
    )
    assert (
        "runtimeClassName" not in k3s[("Deployment", "t-toolrank")]["spec"]["template"]["spec"]
    )  # no GPU there
    k3s = render(
        "auth.apiKey=x",
        "embedding.mode=bundled",
        "embedding.bundled.image=r/i:1",
        "embedding.runtimeClassName=nvidia",
    )
    assert k3s[("Deployment", "t-toolrank")]["spec"]["template"]["spec"]["runtimeClassName"] == "nvidia"


def test_tenants_and_mistakes():
    objs = render(
        "embedding.mode=external",
        "embedding.url=http://e/v1",
        'auth.apiKeys={"team":{"key":"k","sources":["time"]}}',
    )
    c = container(objs)
    assert c["args"][7:9] == ["--api-keys", "/keys/keys.json"] and "TOOLRANK_API_KEY" not in env(c)
    keys = json.loads(objs[("Secret", "t-toolrank-auth")]["stringData"]["keys.json"])
    assert keys == {"team": {"key": "k", "sources": ["time"]}}
    mine = render("embedding.mode=external", "embedding.url=http://e/v1", "auth.apiKeysSecret.name=mine")
    vol = next(
        v
        for v in container(mine) and mine[("Deployment", "t-toolrank")]["spec"]["template"]["spec"]["volumes"]
        if v["name"] == "keys"
    )
    assert vol["secret"] == {"secretName": "mine", "items": [{"key": "keys.json", "path": "keys.json"}]}
    assert ("Secret", "t-toolrank-auth") not in mine

    assert "auth: set auth.apiKey" in render(fails=True)
    assert "embedding.url" in render("auth.apiKey=x", "embedding.mode=external", fails=True)
    assert "embedding.bundled.image" in render("auth.apiKey=x", "embedding.mode=bundled", fails=True)
    assert "embedding.profile" in render("auth.apiKey=x", "embedding.profile=int4", fails=True)
    assert "embedding.mode" in render("auth.apiKey=x", "embedding.mode=cloud", fails=True)
    # values merge: the example MCP server goes only when it is dropped
    only_openapi = render(
        "auth.apiKey=x", 'config={"mcpServers":{"time":null},"openapi":{"api":{"base_url":"https://a"}}}'
    )
    assert json.loads(only_openapi[("ConfigMap", "t-toolrank-config")]["data"]["toolrank.json"]) == {
        "mcpServers": {},
        "openapi": {"api": {"base_url": "https://a"}},
    }
    assert "initContainers" not in only_openapi[("Deployment", "t-toolrank")]["spec"]["template"]["spec"]
