import type { Position } from './window-tracking';

/** Exact critically damped trajectory between CV observations.
 * A new target cannot instantaneously replace the camera velocity. The former
 * first-order lerp was position-continuous but changed velocity on every packet.
 * 1.7 maps the existing response control to a similar 90% settling time.
 */
export class WindowMotion {
  readonly position: Position;
  readonly velocity: Position = { x: 0, y: 0, z: 0 };

  constructor(initial: Position) { this.position = { ...initial }; }

  reset(value: Position): Position {
    Object.assign(this.position, value);
    this.velocity.x = 0;
    this.velocity.y = 0;
    this.velocity.z = 0;
    return this.position;
  }

  update(target: Position, seconds: number, response: number): Position {
    if (![target.x, target.y, target.z, seconds, response].every(Number.isFinite)
        || seconds <= 0 || response <= 0) return this.position;
    const omega = 1.7 * response;
    const decay = Math.exp(-omega * seconds);
    for (const axis of ['x', 'y', 'z'] as const) {
      const offset = this.position[axis] - target[axis];
      const term = (this.velocity[axis] + omega * offset) * seconds;
      this.position[axis] = target[axis] + (offset + term) * decay;
      this.velocity[axis] = (this.velocity[axis] - omega * term) * decay;
    }
    return this.position;
  }
}
