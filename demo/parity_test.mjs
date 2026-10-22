// Parity check: the JS step must match the Python (PyTorch) step.
// Usage: node demo/parity_test.mjs [path/to/fixture.json]   (default: demo/parity_fixture.json)
import { readFileSync } from 'node:fs';
import './nca.js';

const path = process.argv[2] ?? new URL('./parity_fixture.json', import.meta.url);
const fx = JSON.parse(readFileSync(path, 'utf8'));
if (fx.H !== NCA.H || fx.W !== NCA.W) throw new Error(`fixture is ${fx.H}x${fx.W}, nca.js is ${NCA.H}x${NCA.W}`);

const model = NCA.loadModel(fx.model);
let worst = 0;
fx.cases.forEach((cs, n) => {
  const s = Float32Array.from(cs.state_in);
  NCA.step(s, model, 1, cs.angle);
  let err = 0;
  for (let i = 0; i < s.length; i++) err = Math.max(err, Math.abs(s[i] - cs.state_out[i]));
  console.log(`case ${n} (angle ${cs.angle}): max |js - python| = ${err.toExponential(2)}`);
  worst = Math.max(worst, err);
});
console.log(worst <= 1e-4 ? 'PASS' : 'FAIL (tolerance 1e-4)');
process.exit(worst <= 1e-4 ? 0 : 1);
