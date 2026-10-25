import assert from "node:assert/strict";
import { prepareRgba } from "../dist/preprocess.mjs";

function image(width, height, pixel) {
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) rgba.set(pixel(x, y), (y * width + x) * 4);
  }
  return rgba;
}

function checkContract(result) {
  assert.ok(result.tensor instanceof Float32Array);
  assert.ok(result.previewRgba instanceof Uint8ClampedArray);
  assert.equal(result.tensor.length, 784);
  assert.equal(result.previewRgba.length, 3136);
  assert.ok(["light", "dark"].includes(result.backgroundUsed));
  assert.ok(Array.isArray(result.warnings));
  for (let i = 0; i < result.tensor.length; i += 1) {
    const value = result.tensor[i];
    assert.ok(Number.isFinite(value) && value >= 0 && value <= 1);
    assert.ok(Math.abs(value - result.previewRgba[i * 4] / 255) < 1e-7);
    assert.equal(result.previewRgba[i * 4], result.previewRgba[i * 4 + 1]);
    assert.equal(result.previewRgba[i * 4], result.previewRgba[i * 4 + 2]);
    assert.equal(result.previewRgba[i * 4 + 3], 255);
  }
}

const allWhite = prepareRgba(image(16, 12, () => [255, 255, 255, 255]), 16, 12);
checkContract(allWhite);
assert.equal(allWhite.backgroundUsed, "light");
assert.ok(allWhite.tensor.every(value => value === 0));
assert.ok(allWhite.warnings.length >= 2);

const allBlack = prepareRgba(image(16, 12, () => [0, 0, 0, 255]), 16, 12);
checkContract(allBlack);
assert.equal(allBlack.backgroundUsed, "dark");
assert.ok(allBlack.tensor.every(value => value === 0));
assert.ok(allBlack.warnings.length >= 2);

// A dark product rectangle on white: invert, crop its known bounds, center it.
const rectangle = image(40, 30, (x, y) => x >= 13 && x < 23 && y >= 5 && y < 25
  ? [0, 0, 0, 255] : [255, 255, 255, 255]);
const centered = prepareRgba(rectangle, 40, 30);
checkContract(centered);
assert.equal(centered.backgroundUsed, "light");
assert.deepEqual(centered.crop, { x: 13, y: 5, width: 10, height: 20 });
let minX = 28, maxX = -1, minY = 28, maxY = -1;
for (let y = 0; y < 28; y += 1) {
  for (let x = 0; x < 28; x += 1) {
    if (!centered.tensor[y * 28 + x]) continue;
    minX = Math.min(minX, x); maxX = Math.max(maxX, x);
    minY = Math.min(minY, y); maxY = Math.max(maxY, y);
  }
}
assert.deepEqual([minX, minY, maxX, maxY], [8, 2, 19, 25]);
assert.equal((maxX - minX + 1) / (maxY - minY + 1), 0.5);

// The same silhouette with opposite foreground/background gives the same tensor.
const inverseRectangle = image(40, 30, (x, y) => x >= 13 && x < 23 && y >= 5 && y < 25
  ? [255, 255, 255, 255] : [0, 0, 0, 255]);
const dark = prepareRgba(inverseRectangle, 40, 30);
assert.equal(dark.backgroundUsed, "dark");
assert.deepEqual(dark.tensor, centered.tensor);

// Turning centering off keeps the entire source frame, letterboxed without distortion.
const uncentered = prepareRgba(rectangle, 40, 30, { center: false });
checkContract(uncentered);
assert.deepEqual(uncentered.crop, { x: 0, y: 0, width: 40, height: 30 });
assert.notDeepEqual(uncentered.tensor, centered.tensor);
const wide = prepareRgba(image(40, 10, () => [255, 255, 255, 255]), 40, 10,
  { background: "dark", center: false });
assert.equal(wide.tensor.filter(value => value === 1).length, 28 * 7);

// Transparent margins composite as white in auto mode; dark transparent products work.
const transparentRectangle = image(40, 30, (x, y) => x >= 13 && x < 23 && y >= 5 && y < 25
  ? [0, 0, 0, 255] : [120, 70, 200, 0]);
assert.deepEqual(prepareRgba(transparentRectangle, 40, 30).tensor, centered.tensor);
// Explicit dark backdrop makes a white product with alpha usable too.
const transparentWhite = image(40, 30, (x, y) => x >= 13 && x < 23 && y >= 5 && y < 25
  ? [255, 255, 255, 255] : [120, 70, 200, 0]);
assert.deepEqual(prepareRgba(transparentWhite, 40, 30, { background: "dark" }).tensor, centered.tensor);
const transparentEmpty = prepareRgba(image(3, 2, () => [45, 100, 160, 0]), 3, 2);
checkContract(transparentEmpty);
assert.ok(transparentEmpty.tensor.every(value => value === 0));

const gray = prepareRgba(image(10, 10, () => [140, 140, 140, 255]), 10, 10);
assert.ok(gray.warnings.some(value => value.includes("az ton farkı")));
const busy = prepareRgba(image(12, 12, (x, y) => (x + y) % 2 ? [255, 255, 255, 255] : [0, 0, 0, 255]), 12, 12);
assert.ok(busy.warnings.some(value => value.includes("kenar tonları")));

// Color conversion uses luminance, not just the red component or an RGB mean.
const red = prepareRgba(image(28, 28, () => [255, 0, 0, 255]), 28, 28,
  { background: "dark", center: false });
assert.equal(red.previewRgba[0], 76);
checkContract(red);

assert.throws(() => prepareRgba([], 0, 2), TypeError);
assert.throws(() => prepareRgba([], 1.5, 2), TypeError);
assert.throws(() => prepareRgba([], 8193, 1), RangeError);
assert.throws(() => prepareRgba([], 2049, 2049), RangeError);
assert.throws(() => prepareRgba(new Uint8Array(3), 1, 1), TypeError);
assert.throws(() => prepareRgba([0, 0, NaN, 255], 1, 1), RangeError);
assert.throws(() => prepareRgba([0, 0, Infinity, 255], 1, 1), RangeError);
assert.throws(() => prepareRgba([0, 0, 256, 255], 1, 1), RangeError);
assert.throws(() => prepareRgba([0, 0, 0, 255], 1, 1, null), TypeError);
assert.throws(() => prepareRgba([0, 0, 0, 255], 1, 1, { background: "dataset" }), TypeError);
assert.throws(() => prepareRgba([0, 0, 0, 255], 1, 1, { center: "yes" }), TypeError);

// Non-square/single-pixel boundary cases stay finite and preview-identical.
for (const [width, height] of [[1, 1], [1, 13], [19, 1], [31, 17]]) {
  checkContract(prepareRgba(image(width, height, (x, y) => [x % 256, y % 256, 50, 200]), width, height));
}

console.log("PASS: preprocessing input validation, grayscale/inversion, transparent images, exact crop, aspect ratio, centering, quality warnings, and preview/tensor identity.");
