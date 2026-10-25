import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { performance } from 'node:perf_hooks';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { decodeModel, loadModel, runModel } from '../dist/inference.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const metadata = JSON.parse(await readFile(resolve(root, 'dist/model/model.json'), 'utf8'));
const bytes = await readFile(resolve(root, 'dist/model/weights.bin'));
const weights = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
const model = decodeModel(metadata, weights);
const fixture = JSON.parse(await readFile(resolve(root, 'references/runtime-expected.json'), 'utf8'));
function floatArray(buffer) {
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  return Float32Array.from({ length: buffer.byteLength / 4 }, (_, index) => view.getFloat32(index * 4, true));
}
const inputs = floatArray(await readFile(resolve(root, 'references/runtime-inputs.bin')));
const activationReferences = floatArray(await readFile(resolve(root, 'references/runtime-activations.bin')));
let maxProbabilityError = 0, maxLogitError = 0, top1Matches = 0, maxActivationError = 0;
const started = performance.now();
for (let sample = 0; sample < fixture.indices.length; sample++) {
  const result = runModel(model, inputs.subarray(sample * 784, (sample + 1) * 784), { capture: sample === 0 });
  for (let label = 0; label < 10; label++) {
    maxProbabilityError = Math.max(maxProbabilityError, Math.abs(result.probabilities[label] - fixture.probabilities[sample][label]));
    maxLogitError = Math.max(maxLogitError, Math.abs(result.logits[label] - fixture.logits[sample][label]));
  }
  if (result.top3[0].index === fixture.top1[sample]) top1Matches++;
  assert.ok(Math.abs(result.probabilities.reduce((sum, value) => sum + value, 0) - 1) < 1e-12);
  if (sample === 0) {
    for (const check of fixture.activationChecks) {
      const actual = result.activations[check.name];
      assert.deepEqual(actual.shape, check.shape);
      for (let index = 0; index < check.length; index++) {
        maxActivationError = Math.max(maxActivationError,
          Math.abs(actual.data[index] - activationReferences[check.offset + index]));
      }
    }
  }
}
const totalMilliseconds = performance.now() - started;
assert.equal(top1Matches, fixture.indices.length, 'Class predictions must match independent PyTorch references.');
assert.ok(maxProbabilityError < 0.0001, `Probability error ${maxProbabilityError} exceeds declared tolerance.`);
assert.ok(maxLogitError < 0.0001, `Logit error ${maxLogitError} exceeds declared tolerance.`);
assert.ok(maxActivationError < 0.0001, `Activation error ${maxActivationError} exceeds declared tolerance.`);
assert.throws(() => runModel(model, new Float32Array(783)));
const invalid = new Float32Array(784); invalid[0] = NaN;
assert.throws(() => runModel(model, invalid));
invalid[0] = -0.01; assert.throws(() => runModel(model, invalid));
invalid[0] = 1.01; assert.throws(() => runModel(model, invalid));
assert.throws(() => decodeModel(metadata, new ArrayBuffer(4)));
const realFetch = globalThis.fetch;
try {
  globalThis.fetch = async url => String(url).endsWith('model.json')
    ? new Response(JSON.stringify(metadata)) : new Response(bytes);
  const loaded = await loadModel('https://local-runtime-check.invalid/model/');
  assert.equal(loaded.metadata.checkpointSha256, fixture.checkpointSha256);
  if (globalThis.crypto?.subtle) {
    const corrupt = Uint8Array.from(bytes); corrupt[0] ^= 1;
    globalThis.fetch = async url => String(url).endsWith('model.json')
      ? new Response(JSON.stringify(metadata)) : new Response(corrupt);
    await assert.rejects(loadModel('https://local-runtime-check.invalid/model/'), /bütünlük/);
  }
  globalThis.fetch = async () => new Response('', { status: 404 });
  await assert.rejects(loadModel('https://local-runtime-check.invalid/model/'), /yüklenemedi/);
} finally {
  globalThis.fetch = realFetch;
}
const result = {
  checkpointSha256: fixture.checkpointSha256,
  examples: fixture.indices.length, top1Matches,
  absoluteTolerance: 0.0001, maxProbabilityError, maxLogitError, maxActivationError,
  totalMilliseconds, meanMilliseconds: totalMilliseconds / fixture.indices.length,
  loaderChecks: ['successful mocked fetch', 'SHA256 corruption rejected when WebCrypto exists', 'HTTP failure rejected'],
  note: 'Balanced first-ten validation samples per class; independent PyTorch CPU reference. This verifies runtime parity, not accuracy on arbitrary photos.',
};
console.log(JSON.stringify(result, null, 2));
