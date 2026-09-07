"""Deterministic filter comparison. Does not access or store camera images."""
import json
import statistics

from face_tracker.filtering import OneEuroFilter


def compare(beta):
    noise = [0.6 + (0.003 if i % 2 else -0.003) for i in range(180)]
    filter_ = OneEuroFilter(beta=beta)
    filtered = [filter_.apply(p, i / 30) for i, p in enumerate(noise)]
    motion = [0.0] * 30 + [(i + 1) * 0.006 for i in range(30)]
    filter_ = OneEuroFilter(beta=beta)
    tracked = [filter_.apply(p, i / 30) for i, p in enumerate(motion)]
    error = statistics.mean(abs(motion[i] - tracked[i]) for i in range(40, 60))
    return {"beta": beta, "stationary_raw_std_mm": 3.0,
            "stationary_filtered_std_mm": round(statistics.pstdev(filtered[60:]) * 1000, 3),
            "ramp_mean_error_mm": round(error * 1000, 3),
            "ramp_equivalent_lag_ms": round(error / 0.18 * 1000, 2)}


print(json.dumps({"note": "Synthetic 30 Hz; ±3 mm alternating noise; 0.18 m/s ramp. Not physical accuracy or capture-to-photon latency.",
                  "before": compare(0.035), "after": compare(4.0)}, indent=2))
