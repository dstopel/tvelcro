#!/usr/bin/env bash
# Deploy a RunPod GPU pod from the CLI, with a CUDA version constraint.
#
# Why not runpodctl: `runpodctl create pod` has no flag for CUDA version, and
# landing on an old driver breaks current vLLM (needs CUDA 13). The REST API
# accepts allowedCudaVersions, so this script uses that.
#
# Usage:
#   export RUNPOD_API_KEY=rpa_...
#   bash scripts/deploy_pod.sh                          # A5000, CUDA 13 host + CUDA 13 image, no volume
#   GPU="NVIDIA L4" CUDA=12.8 bash scripts/deploy_pod.sh
#   VOLUME=zlrarfve3h DC=EU-RO-1 bash scripts/deploy_pod.sh   # attach the tvelcro-data volume
#   LIST=1 bash scripts/deploy_pod.sh                   # just list running pods
#   STOP=<podid> bash scripts/deploy_pod.sh             # stop a pod
set -euo pipefail

API="https://rest.runpod.io/v1"
KEY="${RUNPOD_API_KEY:?set RUNPOD_API_KEY}"
NAME="${NAME:-play}"
GPU="${GPU:-NVIDIA RTX A5000}"
CUDA="${CUDA:-13.0}"          # valid: 13.0 12.9 12.8 12.7 12.6 12.5 12.4 12.3 12.2 12.1 12.0 11.8
DISK="${DISK:-60}"            # container disk GB; model weights land here when no volume
IMAGE="${IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
VOLUME="${VOLUME:-}"          # network volume id, empty for none
DC="${DC:-}"                  # datacenter id, required when attaching a volume
TEMPLATE="${TEMPLATE:-jrqhr9ll86}"  # default: tvelcro-ml-cu13 (CUDA 13 image, nvcc 13). Auto-runs
                              # /workspace/env/bootstrap.sh when the volume is attached

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

PAYLOAD=$(python3 - <<PY
import json, os
p = {
  "name": os.environ["NAME"],
  "imageName": os.environ["IMAGE"],
  "gpuTypeIds": [os.environ["GPU"]],
  "gpuCount": 1,
  "cloudType": "SECURE",
  "containerDiskInGb": int(os.environ["DISK"]),
  "ports": ["22/tcp", "8888/http"],
  "supportPublicIp": True,
  "allowedCudaVersions": [os.environ["CUDA"]],
}
if os.environ.get("VOLUME"):
    p["networkVolumeId"] = os.environ["VOLUME"]
    p["volumeMountPath"] = "/workspace"
if os.environ.get("DC"):
    p["dataCenterIds"] = [os.environ["DC"]]
if os.environ.get("TEMPLATE"):
    p["templateId"] = os.environ["TEMPLATE"]
print(json.dumps(p))
PY
)

echo "deploying: gpu=$GPU cuda=$CUDA disk=${DISK}GB volume=${VOLUME:-none}"
RESP=$(api -X POST "$API/pods" -d "$PAYLOAD")
POD_ID=$(echo "$RESP" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("id",""))' 2>/dev/null || true)

if [ -z "$POD_ID" ]; then
    echo "FAILED: $RESP"
    echo "hint: no stock for that GPU+CUDA combination. Try another GPU or CUDA=12.8."
    exit 1
fi
echo "pod $POD_ID created; waiting for ssh ..."

for _ in $(seq 1 40); do
    OUT=$(api "$API/pods/$POD_ID" | python3 -c '
import json,sys
p = json.load(sys.stdin)
pm = p.get("portMappings") or {}
print(p.get("publicIp") or "", pm.get("22") or "")' 2>/dev/null || echo "")
    IP=$(echo "$OUT" | cut -d" " -f1); PORT=$(echo "$OUT" | cut -d" " -f2)
    if [ -n "$IP" ] && [ -n "$PORT" ]; then
        echo
        echo "ready:  ssh -i ~/.runpod/ssh/runpodctl-ssh-key -p $PORT root@$IP"
        echo "stop:   STOP=$POD_ID bash scripts/deploy_pod.sh"
        exit 0
    fi
    sleep 15
done
echo "pod $POD_ID created but ssh not exposed yet; check: LIST=1 bash scripts/deploy_pod.sh"
