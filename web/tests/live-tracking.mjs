// Read-only, 90-packet aggregate probe; prints no face images or position history.
const ws = new WebSocket('ws://127.0.0.1:8765/ws/v1/tracking');
const samples = [];
const timeout = setTimeout(() => { console.error('Tracking probe timed out'); ws.close(); process.exitCode = 1; }, 15000);
ws.addEventListener('message', (event) => {
  const data = JSON.parse(event.data);
  samples.push({ time: performance.now(), processing: data.processing_ms,
    age: Date.now() - data.captured_at_unix_ms, tracked: data.tracking, id: data.track_id,
    queue: data.queue_age_ms, captureFps: data.frame.fps, dropped: data.capture_dropped_frames,
    backend: data.tracker_backend, reason: data.quality_reason,
    calibrationReady: typeof data.calibration_ready === 'boolean' ? data.calibration_ready : null,
    reprojection: data.face?.quality?.reprojection_error_px,
    stationary: data.face?.quality?.stationary === true,
    poseSolver: data.face?.quality?.pose_solver,
    yaw: data.face?.head_rotation_deg?.yaw, pitch: data.face?.head_rotation_deg?.pitch });
  if (samples.length < 90) return;
  clearTimeout(timeout);
  ws.close();
  const intervals = samples.slice(1).map((s, i) => s.time - samples[i].time);
  const summary = (values) => {
    const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
    if (!sorted.length) return null;
    return { mean: +(sorted.reduce((a, b) => a + b, 0) / sorted.length).toFixed(2),
      p95: +sorted[Math.floor((sorted.length - 1) * 0.95)].toFixed(2) };
  };
  console.log(JSON.stringify({ packets: samples.length, hz: +(1000 / summary(intervals).mean).toFixed(2),
    interval_ms: summary(intervals), processing_ms: summary(samples.map((s) => s.processing)),
    backend: samples[0].backend,
    camera_capture_fps: summary(samples.map((s) => s.captureFps)),
    queue_age_ms: summary(samples.map((s) => s.queue)),
    dropped_during_probe: samples.at(-1).dropped - samples[0].dropped,
    quality_reasons: samples.reduce((counts, s) => { const key = s.reason ?? 'not_reported'; counts[key] = (counts[key] ?? 0) + 1; return counts; }, {}),
    reprojection_error_px: summary(samples.map((s) => s.reprojection)),
    read_to_receive_ms: summary(samples.slice(1).map((s) => s.age)),
    tracking_ratio: samples.filter((s) => s.tracked).length / samples.length,
    stationary_frames: samples.filter((s) => s.stationary).length,
    pose_solvers: samples.reduce((counts, s) => { const key = s.poseSolver ?? 'not_reported'; counts[key] = (counts[key] ?? 0) + 1; return counts; }, {}),
    calibration_ready_ratio: samples.every((s) => s.calibrationReady === null) ? null
      : samples.filter((s) => s.calibrationReady === true).length / samples.filter((s) => s.calibrationReady !== null).length,
    track_ids: [...new Set(samples.map((s) => s.id))],
    note: 'Read completion to local receiver, excludes camera exposure/driver queue and browser display.' }, null, 2));
});
ws.addEventListener('error', () => { clearTimeout(timeout); console.error('Tracking connection failed'); process.exitCode = 1; });
