#!/usr/bin/env bash
# Regenerate grafana-dashboard-configmap.yaml from the dashboard JSON.
# Run from the repo root after editing deploy/grafana/mlx-memory-dashboard.json.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

python3 - <<'PY'
import textwrap
d = open('deploy/grafana/mlx-memory-dashboard.json').read().rstrip('\n')
body = textwrap.indent(d, '    ')
cm = f"""# Auto-loaded by the kube-prometheus-stack Grafana sidecar (watches all
# namespaces for ConfigMaps labeled grafana_dashboard: "1").
#
# GitOps: copy into homelab-infra apps/kube-prometheus-stack-config/manifests/.
# If you edit the dashboard, regenerate this file:
#   deploy/k8s/gen-dashboard-configmap.sh
apiVersion: v1
kind: ConfigMap
metadata:
  name: grafana-dashboard-mlx-memory
  namespace: monitoring
  labels:
    grafana_dashboard: "1"
  annotations:
    grafana_folder: "Homelab"
data:
  mlx-memory-dashboard.json: |
{body}
"""
open('deploy/k8s/grafana-dashboard-configmap.yaml', 'w').write(cm)
print("regenerated deploy/k8s/grafana-dashboard-configmap.yaml")
PY
