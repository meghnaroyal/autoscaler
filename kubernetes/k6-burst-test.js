/**
 * k6-burst-test.js
 *
 * Bursty load test for comparing GRU autoscaler vs Kubernetes HPA.
 *
 * Pattern (total ~7 min):
 *   30s  — baseline    (20 VUs)   → steady moderate CPU
 *    5s  — spike #1   (300 VUs)   → instant 15x surge
 *   60s  — sustain     (300 VUs)  → HPA scrambles to catch up
 *    5s  — drop #1     (5 VUs)    → instant near-zero
 *   60s  — recovery    (20 VUs)   → HPA holds replicas (300s window), GRU releases sooner
 *    5s  — spike #2   (300 VUs)   → second burst (tests if GRU learned from spike #1)
 *   60s  — sustain     (300 VUs)
 *    5s  — drop #2     (5 VUs)
 *   30s  — cooldown    (20 VUs)
 *
 * Why this exposes HPA lag:
 *   - HPA scrapes CPU every 15 s and needs ~30-45 s to provision new pods.
 *     During that window all 300 VUs hammer the existing (under-provisioned) pods.
 *   - GRU sees the rising CPU history and scales up *before* the spike fully lands.
 *   - On drop: HPA's stabilizationWindowSeconds=300 keeps excess replicas for 5 min.
 *     GRU follows the predicted CPU down and releases replicas sooner.
 *
 * Usage (inside the cluster via loadgen-k6-burst.yaml Job):
 *   kubectl apply -f kubernetes/loadgen-k6-burst.yaml
 *
 * Usage (local, port-forward required):
 *   kubectl port-forward svc/cpu-load-app 8000:8000 -n ai-scaler
 *   k6 run --env BASE_URL=http://localhost:8000 kubernetes/k6-burst-test.js
 */

import http from 'k6/http';
import { sleep, check } from 'k6';

// Override BASE_URL via --env flag or the Job env var for in-cluster runs
const BASE_URL = __ENV.BASE_URL || 'http://cpu-load-app.ai-scaler:8000';

export const options = {
  stages: [
    // Baseline — give autoscalers a stable reference window
    { duration: '30s', target: 20 },

    // Spike #1 — instant 15x burst (5 s ramp ≈ immediate)
    { duration: '5s',  target: 300 },
    { duration: '60s', target: 300 },

    // Drop #1 — instant near-zero
    { duration: '5s',  target: 5 },
    { duration: '60s', target: 20 },

    // Spike #2 — second burst to test GRU's pattern memory
    { duration: '5s',  target: 300 },
    { duration: '60s', target: 300 },

    // Drop #2 — instant near-zero again
    { duration: '5s',  target: 5 },

    // Cooldown
    { duration: '30s', target: 20 },
  ],

  // Fail the run if error rate exceeds 10 % (indicates severe under-provisioning)
  thresholds: {
    http_req_failed: ['rate<0.10'],
  },
};

export default function () {
  // Each VU fires one CPU-intensive request; iterations value chosen so a
  // single pod (~0.1 CPU request, 1 CPU limit) saturates at ~50-60 concurrent VUs.
  const res = http.get(`${BASE_URL}/work?iterations=5000000`);

  check(res, {
    'status 200': (r) => r.status === 200,
  });

  // No sleep — we want maximum concurrency pressure per VU during spikes
}
