/**
 * Prepare a simple product photo for the Fashion-MNIST input contract.
 *
 * This is an explicit grayscale/background/centering heuristic, not semantic
 * background removal. Dataset examples must bypass it and keep their original
 * 28 x 28 pixels. It cannot make arbitrary photographs match the training data.
 */

const OUTPUT_SIDE = 28;
const MAX_PIXELS = 4_194_304;
const BACKGROUNDS = new Set(["auto", "light", "dark"]);

function validateInput(rgba, width, height, options) {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1) {
    throw new TypeError("Görüntü boyutları pozitif tam sayılar olmalı.");
  }
  if (width > 8192 || height > 8192 || width * height > MAX_PIXELS) {
    throw new RangeError("Görüntü çok büyük; önce en uzun kenarını 1024 piksele küçültün.");
  }
  if ((!Array.isArray(rgba) && !ArrayBuffer.isView(rgba)) || rgba.length !== width * height * 4) {
    throw new TypeError("RGBA verisi görüntü boyutlarıyla eşleşmiyor.");
  }
  if (options === null || typeof options !== "object" || Array.isArray(options)) {
    throw new TypeError("Ön işleme seçenekleri bir nesne olmalı.");
  }
  const background = options.background === undefined ? "auto" : options.background;
  const center = options.center === undefined ? true : options.center;
  if (!BACKGROUNDS.has(background)) {
    throw new TypeError("Zemin seçimi auto, light veya dark olmalı.");
  }
  if (typeof center !== "boolean") {
    throw new TypeError("Ortalamayı açma/kapama seçeneği boolean olmalı.");
  }
  for (let i = 0; i < rgba.length; i += 1) {
    if (typeof rgba[i] !== "number" || !Number.isFinite(rgba[i]) || rgba[i] < 0 || rgba[i] > 255) {
      throw new RangeError("RGBA kanal değerleri sonlu ve 0–255 aralığında olmalı.");
    }
  }
  return { background, center };
}

function luminance(red, green, blue) {
  return 0.299 * red + 0.587 * green + 0.114 * blue;
}

function inspectBorder(rgba, width, height) {
  let sum = 0;
  let squareSum = 0;
  let count = 0;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (x !== 0 && y !== 0 && x !== width - 1 && y !== height - 1) continue;
      const offset = (y * width + x) * 4;
      const alpha = rgba[offset + 3] / 255;
      // Transparent border pixels count as white for the automatic decision.
      const value = luminance(rgba[offset], rgba[offset + 1], rgba[offset + 2]) * alpha
        + 255 * (1 - alpha);
      sum += value;
      squareSum += value * value;
      count += 1;
    }
  }
  const mean = sum / count;
  return { mean, deviation: Math.sqrt(Math.max(0, squareSum / count - mean * mean)) };
}

function toForegroundBrightness(rgba, width, height, backgroundUsed) {
  const values = new Float32Array(width * height);
  const backdrop = backgroundUsed === "light" ? 255 : 0;
  let minimum = 255;
  let maximum = 0;
  for (let i = 0; i < values.length; i += 1) {
    const offset = i * 4;
    const alpha = rgba[offset + 3] / 255;
    const gray = luminance(rgba[offset], rgba[offset + 1], rgba[offset + 2]) * alpha
      + backdrop * (1 - alpha);
    const value = Math.max(0, Math.min(255, backgroundUsed === "light" ? 255 - gray : gray));
    values[i] = value;
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
  }
  return { values, minimum, maximum };
}

function foregroundBox(values, width, height, maximum) {
  // Small compression noise should not define the product's bounding box.
  // This threshold is not an object detector and may include a busy background.
  const threshold = Math.max(12, maximum * 0.12);
  let left = width;
  let top = height;
  let right = -1;
  let bottom = -1;
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      if (values[y * width + x] < threshold) continue;
      left = Math.min(left, x);
      right = Math.max(right, x);
      top = Math.min(top, y);
      bottom = Math.max(bottom, y);
    }
  }
  return right < left ? null : { x: left, y: top, width: right - left + 1, height: bottom - top + 1 };
}

function resizeIntoSquare(values, sourceWidth, crop, margin) {
  const available = OUTPUT_SIDE - margin * 2;
  const scale = Math.min(available / crop.width, available / crop.height);
  const targetWidth = Math.max(1, Math.min(available, Math.round(crop.width * scale)));
  const targetHeight = Math.max(1, Math.min(available, Math.round(crop.height * scale)));
  const left = Math.floor((OUTPUT_SIDE - targetWidth) / 2);
  const top = Math.floor((OUTPUT_SIDE - targetHeight) / 2);
  const pixels = new Uint8Array(OUTPUT_SIDE * OUTPUT_SIDE);

  // Each output pixel is an area-weighted average of source pixel boxes.
  // This is a simple anti-aliased resize, with black letterboxing around it.
  for (let dy = 0; dy < targetHeight; dy += 1) {
    const y0 = crop.y + dy * crop.height / targetHeight;
    const y1 = crop.y + (dy + 1) * crop.height / targetHeight;
    for (let dx = 0; dx < targetWidth; dx += 1) {
      const x0 = crop.x + dx * crop.width / targetWidth;
      const x1 = crop.x + (dx + 1) * crop.width / targetWidth;
      let sum = 0;
      for (let sy = Math.floor(y0); sy < Math.min(crop.y + crop.height, Math.ceil(y1)); sy += 1) {
        const overlapY = Math.max(0, Math.min(y1, sy + 1) - Math.max(y0, sy));
        for (let sx = Math.floor(x0); sx < Math.min(crop.x + crop.width, Math.ceil(x1)); sx += 1) {
          const overlapX = Math.max(0, Math.min(x1, sx + 1) - Math.max(x0, sx));
          sum += values[sy * sourceWidth + sx] * overlapX * overlapY;
        }
      }
      const average = sum / ((x1 - x0) * (y1 - y0));
      pixels[(top + dy) * OUTPUT_SIDE + left + dx] = Math.round(Math.max(0, Math.min(255, average)));
    }
  }
  return pixels;
}

/**
 * @param {ArrayLike<number>} rgba Unpremultiplied RGBA, e.g. ImageData.data.
 * @param {number} width
 * @param {number} height
 * @param {{background?:'auto'|'light'|'dark', center?:boolean}} options
 * @returns {{tensor:Float32Array,previewRgba:Uint8ClampedArray,backgroundUsed:string,crop:object,warnings:string[]}}
 */
export function prepareRgba(rgba, width, height, options = {}) {
  const selected = validateInput(rgba, width, height, options);
  const border = inspectBorder(rgba, width, height);
  const backgroundUsed = selected.background === "auto"
    ? (border.mean >= 127.5 ? "light" : "dark") : selected.background;
  const gray = toForegroundBrightness(rgba, width, height, backgroundUsed);
  const detected = foregroundBox(gray.values, width, height, gray.maximum);
  const fullImage = { x: 0, y: 0, width, height };
  const crop = selected.center && detected ? detected : fullImage;
  const warnings = [];
  if (gray.maximum - gray.minimum < 20) {
    warnings.push("Görüntüde az ton farkı var; ürün ve zemin arasında daha belirgin kontrast deneyin.");
  }
  if (!detected) {
    warnings.push("Zeminden ayrışan bir ürün bulunamadı; görüntünün tamamı kullanıldı.");
  }
  if (border.deviation > 45) {
    warnings.push("Fotoğrafın kenar tonları değişken; sade ve tek renk bir zemin daha uygun olabilir.");
  }
  const pixels = resizeIntoSquare(gray.values, width, crop, selected.center ? 2 : 0);
  const tensor = new Float32Array(OUTPUT_SIDE * OUTPUT_SIDE);
  const previewRgba = new Uint8ClampedArray(OUTPUT_SIDE * OUTPUT_SIDE * 4);
  for (let i = 0; i < pixels.length; i += 1) {
    tensor[i] = pixels[i] / 255;
    previewRgba[i * 4] = pixels[i];
    previewRgba[i * 4 + 1] = pixels[i];
    previewRgba[i * 4 + 2] = pixels[i];
    previewRgba[i * 4 + 3] = 255;
  }
  return { tensor, previewRgba, backgroundUsed, crop, warnings };
}
