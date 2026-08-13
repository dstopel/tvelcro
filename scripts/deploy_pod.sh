#!/usr/bin/env bash
# Deploy a RunPod GPU pod from the CLI, with a CUDA version constraint.
#
# Why not runpodctl: `runpodctl create pod` has no flag for CUDA version, and
# landing on an old driver breaks current vLLM (needs CUDA 13). The REST API
# accepts allowedCudaVersions, so this script uses that.
#
# Usage:
#   export RUNPOD_API_KEY=rpa_...
#   bash scripts/deploy_pod.sh                # CUDA 13, tvelcro-data volume at /workspace,
#                                             # cheapest candidate GPU under MAXPRICE
#   MAXPRICE=0.80 bash scripts/deploy_pod.sh  # widen the GPU search
#   VOLUME= DC= bash scripts/deploy_pod.sh    # no volume, any datacenter (more GPU choice)
#   GPU="NVIDIA L40S" bash scripts/deploy_pod.sh        # pin one GPU
#   LIST=1 bash scripts/deploy_pod.sh                   # just list running pods
#   STOP=<podid> bash scripts/deploy_pod.sh             # stop a pod
#
# The volume is datacenter-locked to EU-RO-1, so attaching it restricts which GPUs
# can be had at all. If every candidate fails, that constraint is usually why.
#
# On success it rewrites a managed "Host tvelcro-pod" block in ~/.ssh/config, so
# `ssh tvelcro-pod` and VS Code Remote-SSH keep working across redeploys without
# editing anything. Only the text between the marker comments is touched.
set -euo pipefail

API="https://rest.runpod.io/v1"
KEY="${RUNPOD_API_KEY:?set RUNPOD_API_KEY}"
NAME="${NAME:-play}"
GPU="${GPU:-}"                # pin one GPU; empty means walk CANDIDATES cheapest-first
MAXPRICE="${MAXPRICE:-0.60}"  # $/hr cap applied to CANDIDATES. 0.50 is too tight once the
                              # volume pins us to EU-RO-1: nothing under 0.57 had stock there
CUDA="${CUDA:-13.0}"          # valid: 13.0 12.9 12.8 12.7 12.6 12.5 12.4 12.3 12.2 12.1 12.0 11.8
DISK="${DISK:-60}"            # container disk GB; model weights land here when no volume
IMAGE="${IMAGE:-}"            # empty: the template supplies the image. Set only to override it
VOLUME="${VOLUME:-zlrarfve3h}"  # tvelcro-data, mounted at /workspace. VOLUME= for none
DC="${DC:-EU-RO-1}"           # must match the volume's datacenter, so this pins GPU choice
TEMPLATE="${TEMPLATE:-jrqhr9ll86}"  # default: tvelcro-ml-cu13 (CUDA 13 image, nvcc 13). Auto-runs
                              # /workspace/env/bootstrap.sh when the volume is attached
export NAME GPU CUDA DISK IMAGE VOLUME DC TEMPLATE   # the payload heredoc reads these via os.environ

# Secure-cloud candidates, cheapest first ($/hr, checked 2026-08-13). Listed by price,
# NOT by current stock: the loop below just tries the next one when a card is out.
# All are >=24 GB, so the 7B-in-fp16 exercises fit (A4500 at $0.25 is 20 GB: excluded).
# VRAM: A5000 24, A40 48, 3090 24, A6000 48, PRO4000 24, PRO4500 32, 4090 24, L40S 48, 5090 32
CANDIDATES="0.27 NVIDIA RTX A5000
0.44 NVIDIA A40
0.50 NVIDIA GeForce RTX 3090
0.53 NVIDIA RTX A6000
0.57 NVIDIA RTX PRO 4000 Blackwell
0.72 NVIDIA RTX PRO 4500 Blackwell
0.74 NVIDIA GeForce RTX 4090
0.99 NVIDIA L40S
0.99 NVIDIA GeForce RTX 5090"

api() { curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' "$@"; }

if [ -n "${LIST:-}" ]; then
    api "$API/pods" | python3 -c '
import json, sys
for p in json.load(sys.stdin):
    pm = p.get("portMappings") or {}
    gpu = (p.get("machine") or {}).get("gpuTypeId", "?")
    print("%s  %-16s %-9s %-22s ssh: root@%s -p %s" % (
        p["id"], p["name"], p.get("desiredStatus"), gpu,
        p.get("publicIp", "-"), pm.get("22", "-")))'
    exit 0
fi

if [ -n "${STOP:-}" ]; then
    api -X POST "$API/pods/$STOP/stop" >/dev/null && echo "stopped $STOP"
    exit 0
fi

payload() {   # $1 = gpu type id
  GPU="$1" python3 - <<PY
import json, os

# Runs at pod boot. Backgrounded so sshd is up immediately; the install continues
# while you log in. Progress: tail -f /var/log/pod_setup.log
SETUP = """
echo '[ -f /var/log/pod_setup.done ] && echo "env ready: /opt/mlenv active, data at /workspace" || { [ -f /var/log/pod_setup.failed ] && echo "SETUP FAILED: tail -40 /var/log/pod_setup.log" || echo "setup running: tail -f /var/log/pod_setup.log"; }' >> /root/.bashrc
{ export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq zsh vim tmux htop ripgrep
  [ -x /workspace/env/bootstrap.sh ] && bash /workspace/env/bootstrap.sh
  # venv, never the system python: vllm pulls deps that collide with debian-managed
  # packages (PyJWT has no RECORD file), which aborts pip mid-uninstall and leaves
  # the image's torch broken. --system-site-packages reuses the image torch.
  # Local to the container, NOT on /workspace: the volume's mlenv is cu126 and shared
  # with the tvelcro-gpu experiment pod, so installing there would disturb a running job.
  /opt/mlenv/bin/python -c "import vllm" 2>/dev/null && touch /var/log/pod_setup.done
  if [ ! -f /var/log/pod_setup.done ]; then
    python3 -m venv --system-site-packages /opt/mlenv
    /opt/mlenv/bin/pip install --upgrade pip
    if /opt/mlenv/bin/pip install vllm; then touch /var/log/pod_setup.done
    else touch /var/log/pod_setup.failed; fi
  fi
  # Both rc files, appended after bootstrap so /opt/mlenv wins. bootstrap symlinks
  # /root/.zshrc onto the volume, whose zshrc activates /workspace/mlenv (no torch);
  # replace that symlink with a real file that sources it first, then our venv. Writing
  # to /root keeps this container-local: the shared volume is never modified.
  echo 'source /opt/mlenv/bin/activate 2>/dev/null' >> /root/.bashrc
  if [ -L /root/.zshrc ]; then
    ZSRC=$(readlink -f /root/.zshrc); rm -f /root/.zshrc
    printf 'source %s\nsource /opt/mlenv/bin/activate 2>/dev/null\n' "$ZSRC" > /root/.zshrc
  fi
} > /var/log/pod_setup.log 2>&1 &
/start.sh
"""

p = {
  "dockerStartCmd": ["bash", "-c", SETUP],
  "name": os.environ["NAME"],
  "gpuTypeIds": [os.environ["GPU"]],
  "gpuCount": 1,
  "cloudType": "SECURE",
  "containerDiskInGb": int(os.environ["DISK"]),
  "ports": ["22/tcp", "8888/http"],
  "supportPublicIp": True,
  "allowedCudaVersions": [os.environ["CUDA"]],
}
if os.environ.get("IMAGE"):
    p["imageName"] = os.environ["IMAGE"]
if os.environ.get("VOLUME"):
    p["networkVolumeId"] = os.environ["VOLUME"]
    p["volumeMountPath"] = "/workspace"
if os.environ.get("DC"):
    p["dataCenterIds"] = [os.environ["DC"]]
if os.environ.get("TEMPLATE"):
    p["templateId"] = os.environ["TEMPLATE"]
print(json.dumps(p))
PY
}

# GPU= pins one; otherwise walk CANDIDATES cheapest-first, skipping anything over MAXPRICE.
if [ -n "$GPU" ]; then
    TRY="0.00 $GPU"
else
    TRY=$(echo "$CANDIDATES" | awk -v m="$MAXPRICE" '$1 <= m')
    [ -n "$TRY" ] || { echo "no candidate GPU at or under \$$MAXPRICE/hr; raise MAXPRICE"; exit 1; }
fi

POD_ID=""
while read -r PRICE GPUNAME; do
    [ -n "$GPUNAME" ] || continue
    echo "trying: $GPUNAME (\$$PRICE/hr) cuda=$CUDA disk=${DISK}GB volume=${VOLUME:-none}"
    RESP=$(api -X POST "$API/pods" -d "$(payload "$GPUNAME")" || true)
    POD_ID=$(echo "$RESP" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("id",""))' 2>/dev/null || true)
    [ -n "$POD_ID" ] && { GPU="$GPUNAME"; break; }
    echo "  unavailable: $(echo "$RESP" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("error",""))' 2>/dev/null || echo "$RESP")"
done <<EOF
$TRY
EOF

if [ -z "$POD_ID" ]; then
    echo "FAILED: no stock on CUDA $CUDA for $(echo "$TRY" | wc -l | tr -d ' ') candidate(s) tried."
    echo "hint: raise MAXPRICE (now \$$MAXPRICE) to widen the list, or CUDA=12.8 which has far more stock."
    exit 1
fi
echo "pod $POD_ID created on $GPU; waiting for ssh ..."

for _ in $(seq 1 40); do
    OUT=$(api "$API/pods/$POD_ID" | python3 -c '
import json,sys
p = json.load(sys.stdin)
pm = p.get("portMappings") or {}
print(p.get("publicIp") or "", pm.get("22") or "")' 2>/dev/null || echo "")
    IP=$(echo "$OUT" | cut -d" " -f1); PORT=$(echo "$OUT" | cut -d" " -f2)
    if [ -n "$IP" ] && [ -n "$PORT" ]; then
        # Rewrite a managed block in ~/.ssh/config so the alias is stable across pods:
        # VS Code / ssh always target "tvelcro-pod", never a hardcoded ip:port.
        IP="$IP" PORT="$PORT" python3 - <<'PY'
import os, pathlib
alias = "tvelcro-" + os.environ["NAME"]     # one block per pod name, so parallel pods coexist
cfg = pathlib.Path.home() / ".ssh" / "config"
top, bot = f"# >>> {alias} (managed by deploy_pod.sh) >>>", f"# <<< {alias} <<<"
block = f"""{top}
Host {alias}
    HostName {os.environ['IP']}
    Port {os.environ['PORT']}
    User root
    IdentityFile ~/.runpod/ssh/runpodctl-ssh-key
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
{bot}"""
cfg.parent.mkdir(mode=0o700, exist_ok=True)
old = cfg.read_text() if cfg.exists() else ""
if top in old and bot in old:                      # replace only our block
    new = old[:old.index(top)] + block + old[old.index(bot) + len(bot):]
else:
    new = (old.rstrip() + "\n\n" if old.strip() else "") + block + "\n"
cfg.write_text(new)
cfg.chmod(0o600)
print("~/.ssh/config: %s -> %s:%s" % (alias, os.environ["IP"], os.environ["PORT"]))
PY
        echo
        echo "ready:  ssh tvelcro-$NAME"
        echo "        (or: ssh -i ~/.runpod/ssh/runpodctl-ssh-key -p $PORT root@$IP)"
        echo "vscode: Remote-SSH: Connect to Host -> tvelcro-$NAME, then open /workspace"
        echo "stop:   STOP=$POD_ID bash scripts/deploy_pod.sh"
        exit 0
    fi
    sleep 15
done
echo "pod $POD_ID created but ssh not exposed yet; check: LIST=1 bash scripts/deploy_pod.sh"
