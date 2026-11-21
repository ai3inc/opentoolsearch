#!/bin/bash
# Sync the repository (code, scripts, benchmark data) to a pod.  POD=ots-h100 scripts/sync_pod.sh
set -e
cd "$(dirname "$0")/.."
T=$(mktemp -t ots).tgz; tar czf "$T" opentoolsearch scripts benchmarks examples hf data/v3/bench.json data/v4/bench_public.json
onpod cp "$T" ${POD:?}:/workspace/ots.tgz | tail -1
onpod exec ${POD:?} -- bash -lc 'mkdir -p /workspace/ots && cd /workspace/ots && tar xzf /workspace/ots.tgz && mkdir -p /workspace/run && echo synced'
