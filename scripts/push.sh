#!/bin/bash
# Sync code to the training pod.
set -e
cd "$(dirname "$0")/.."
T=$(mktemp -t code).tgz
tar czf "$T" opentoolsearch scripts examples
onpod cp "$T" ${POD:-toolsift-train}:/workspace/code.tgz | tail -1
onpod exec ${POD:-toolsift-train} -- bash -lc 'cd /workspace/toolsift && tar xzf /workspace/code.tgz && echo synced'
