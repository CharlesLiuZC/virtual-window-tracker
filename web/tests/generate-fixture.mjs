// Generate a deterministic binary PLY test asset, not a captured/reconstructed scene.
import { readFileSync, writeFileSync } from 'node:fs';
const source = readFileSync(new URL('./fixtures/gaussian-cube-ascii.ply', import.meta.url), 'utf8');
const [header, body] = source.split('end_header\n');
const values = body.trim().split(/\s+/).map(Number);
const payload = Buffer.alloc(values.length * 4);
values.forEach((value, index) => payload.writeFloatLE(value, index * 4));
const output = new URL('./fixtures/gaussian-cube-binary.ply', import.meta.url);
writeFileSync(output, Buffer.concat([Buffer.from(header.replace('format ascii', 'format binary_little_endian') + 'end_header\n'), payload]));
console.log(`Generated ${values.length / 14} Gaussian fixture: ${output.pathname}`);
