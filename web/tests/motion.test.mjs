import test from 'node:test';
import assert from 'node:assert/strict';
import { WindowMotion } from '../lib/window-motion.ts';

test('constant viewer cannot move the camera', () => {
  const initial = { x: .2, y: -.1, z: 7.4 };
  const motion = new WindowMotion(initial);
  for (let i = 0; i < 200; i++) assert.deepEqual(motion.update(initial, 1 / 60, 28), initial);
});

test('a new observation does not instantaneously jump camera velocity', () => {
  const motion = new WindowMotion({ x: 0, y: 0, z: 7.4 });
  motion.update({ x: 1, y: 0, z: 7.4 }, 1e-6, 28);
  assert.ok(motion.position.x < 1e-8); // O(dt^2), not the former O(dt) lerp step.
  assert.ok(motion.velocity.x < .003);
});

test('trajectory is frame-rate independent for the same held observation', () => {
  const run = hz => {
    const motion = new WindowMotion({ x: 0, y: 0, z: 7.4 });
    for (let i = 0; i < hz / 10; i++) motion.update({ x: 1, y: 1, z: 8 }, 1 / hz, 28);
    return motion.position;
  };
  const a = run(30), b = run(60), c = run(120);
  for (const axis of ['x', 'y', 'z']) {
    assert.ok(Math.abs(a[axis] - b[axis]) < 1e-12);
    assert.ok(Math.abs(b[axis] - c[axis]) < 1e-12);
  }
});

test('step settles promptly without overshoot or freezing motion', () => {
  const motion = new WindowMotion({ x: 0, y: 0, z: 7.4 });
  let previous = 0;
  for (let i = 0; i < 30; i++) {
    const current = motion.update({ x: 1, y: 0, z: 7.4 }, 1 / 60, 28).x;
    assert.ok(current >= previous && current <= 1);
    if (i === 5) assert.ok(current > .9);
    previous = current;
  }
});

test('invalid time and measurements cannot poison the camera', () => {
  const initial = { x: 0, y: 0, z: 7.4 };
  const motion = new WindowMotion(initial);
  assert.deepEqual(motion.update({ ...initial, x: NaN }, 1 / 60, 28), initial);
  assert.deepEqual(motion.update({ ...initial, x: 1 }, -1, 28), initial);
  assert.deepEqual(motion.update({ ...initial, x: 1 }, 1 / 60, Infinity), initial);
});
