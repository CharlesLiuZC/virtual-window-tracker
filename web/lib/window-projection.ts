import type { PerspectiveCamera } from 'three';
import type { Position } from './window-tracking';

/** Screen is the fixed world plane z=0. Never aim the camera at the object. */
export function applyWindowProjection(camera: PerspectiveCamera, eye: Position,
                                      width: number, height: number, near: number, far: number) {
  if (![eye.x, eye.y, eye.z, width, height, near, far].every(Number.isFinite)
      || width <= 0 || height <= 0 || near <= 0 || far <= near) {
    throw new Error('Invalid window projection parameters');
  }
  const distance = Math.max(eye.z, near * 1.01);
  // Clamp the actual eye AND the frustum together to keep all four corners anchored.
  camera.position.set(eye.x, eye.y, distance);
  camera.quaternion.identity();
  camera.near = near;
  camera.far = far;
  camera.zoom = 1;
  camera.aspect = width / height;
  camera.fov = 2 * Math.atan(height / (2 * distance)) * 180 / Math.PI;
  const scale = near / distance;
  camera.projectionMatrix.makePerspective(
    (-width / 2 - eye.x) * scale, (width / 2 - eye.x) * scale,
    (height / 2 - eye.y) * scale, (-height / 2 - eye.y) * scale, near, far,
  );
  camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
  camera.updateMatrixWorld();
  // fov/aspect agree with P00/P11 for downstream culling/LOD. Do not call
  // updateProjectionMatrix afterwards: it would erase the off-axis center.
}
