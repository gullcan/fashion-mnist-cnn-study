/** Real forward pass for the fixed PyTorch CNN. No image or prediction API calls. */

const EXPECTED_SHAPES = {
  '0.weight': [16, 1, 5, 5], '0.bias': [16],
  '3.weight': [32, 16, 5, 5], '3.bias': [32],
  '7.weight': [84, 512], '7.bias': [84],
  '9.weight': [10, 84], '9.bias': [10],
};

export function decodeModel(metadata, arrayBuffer) {
  if (metadata?.format !== 'cnn-learning-browser-v1' || metadata.dtype !== 'float32-le') {
    throw new Error('Model dosyasının biçimi desteklenmiyor.');
  }
  if (!(arrayBuffer instanceof ArrayBuffer) || arrayBuffer.byteLength !== 57190 * 4 ||
      metadata.parameterCount !== 57190 || metadata.weightsBytes !== arrayBuffer.byteLength ||
      !Array.isArray(metadata.classes) || metadata.classes.length !== 10) {
    throw new Error('Model ağırlıkları eksik veya boyutları beklenenden farklı.');
  }
  const view = new DataView(arrayBuffer);
  const parameters = {};
  let expectedOffset = 0;
  for (const [name, shape] of Object.entries(EXPECTED_SHAPES)) {
    const spec = metadata.tensors?.[name];
    const length = shape.reduce((product, value) => product * value, 1);
    if (!spec || JSON.stringify(spec.shape) !== JSON.stringify(shape) ||
        spec.offset !== expectedOffset || spec.length !== length) {
      throw new Error(`Model katmanı uyumsuz: ${name}`);
    }
    const values = new Float32Array(length);
    for (let index = 0; index < length; index++) {
      values[index] = view.getFloat32((spec.offset + index) * 4, true);
      if (!Number.isFinite(values[index])) throw new Error('Model sonlu olmayan katsayı içeriyor.');
    }
    parameters[name] = values;
    expectedOffset += length;
  }
  return { metadata, parameters };
}

export async function loadModel(baseUrl = './model/') {
  const base = new URL(baseUrl, globalThis.document?.baseURI ?? globalThis.location?.href ?? 'http://localhost/');
  const metadataResponse = await fetch(new URL('model.json', base));
  if (!metadataResponse.ok) throw new Error('Model bilgileri yüklenemedi. Sayfayı yenileyebilirsin.');
  const metadata = await metadataResponse.json();
  const weightsResponse = await fetch(new URL('weights.bin', base));
  if (!weightsResponse.ok) throw new Error('Eğitilmiş model ağırlıkları yüklenemedi.');
  const weights = await weightsResponse.arrayBuffer();
  if (globalThis.crypto?.subtle) {
    const digest = await globalThis.crypto.subtle.digest('SHA-256', weights);
    const actualHash = Array.from(new Uint8Array(digest), value => value.toString(16).padStart(2, '0')).join('');
    if (actualHash !== metadata.weightsSha256) throw new Error('Model bütünlük kontrolü başarısız oldu.');
  }
  return decodeModel(metadata, weights);
}

function conv5(input, inChannels, height, width, outChannels, weights, bias) {
  const outputHeight = height - 4, outputWidth = width - 4;
  const output = new Float32Array(outChannels * outputHeight * outputWidth);
  for (let out = 0; out < outChannels; out++) {
    const outputOffset = out * outputHeight * outputWidth;
    for (let row = 0; row < outputHeight; row++) {
      for (let col = 0; col < outputWidth; col++) {
        let sum = bias[out];
        for (let channel = 0; channel < inChannels; channel++) {
          const inputOffset = channel * height * width + row * width + col;
          const weightOffset = (out * inChannels + channel) * 25;
          for (let kernelRow = 0; kernelRow < 5; kernelRow++) {
            const pixelRow = inputOffset + kernelRow * width;
            const weightRow = weightOffset + kernelRow * 5;
            for (let kernelCol = 0; kernelCol < 5; kernelCol++) {
              sum += input[pixelRow + kernelCol] * weights[weightRow + kernelCol];
            }
          }
        }
        output[outputOffset + row * outputWidth + col] = sum;
      }
    }
  }
  return output;
}

function relu(input) {
  const result = new Float32Array(input.length);
  for (let index = 0; index < input.length; index++) result[index] = Math.max(0, input[index]);
  return result;
}

function averagePool2(input, channels, height, width) {
  const outputHeight = height / 2, outputWidth = width / 2;
  const output = new Float32Array(channels * outputHeight * outputWidth);
  for (let channel = 0; channel < channels; channel++) {
    for (let row = 0; row < outputHeight; row++) {
      for (let col = 0; col < outputWidth; col++) {
        const index = channel * height * width + row * 2 * width + col * 2;
        output[channel * outputHeight * outputWidth + row * outputWidth + col] =
          (input[index] + input[index + 1] + input[index + width] + input[index + width + 1]) / 4;
      }
    }
  }
  return output;
}

function linear(input, outFeatures, weights, bias) {
  const output = new Float32Array(outFeatures);
  for (let out = 0; out < outFeatures; out++) {
    let sum = bias[out];
    const weightOffset = out * input.length;
    for (let index = 0; index < input.length; index++) sum += weights[weightOffset + index] * input[index];
    output[out] = sum;
  }
  return output;
}

function softmax(logits) {
  const maximum = Math.max(...logits);
  const exponentials = Array.from(logits, value => Math.exp(value - maximum));
  const sum = exponentials.reduce((total, value) => total + value, 0);
  return exponentials.map(value => value / sum);
}

export function runModel(model, inputFloat32, { capture = false } = {}) {
  if (!model?.parameters || !inputFloat32 || inputFloat32.length !== 784) {
    throw new Error('Model girdisi 28 × 28, yani 784 parlaklık değeri içermeli.');
  }
  for (const value of inputFloat32) {
    if (!Number.isFinite(value) || value < 0 || value > 1) {
      throw new Error('Piksel değerleri sonlu ve 0–1 aralığında olmalı.');
    }
  }
  const p = model.parameters;
  const conv1 = conv5(inputFloat32, 1, 28, 28, 16, p['0.weight'], p['0.bias']);
  const relu1 = relu(conv1);
  const pool1 = averagePool2(relu1, 16, 24, 24);
  const conv2 = conv5(pool1, 16, 12, 12, 32, p['3.weight'], p['3.bias']);
  const relu2 = relu(conv2);
  const pool2 = averagePool2(relu2, 32, 8, 8);
  const hidden = relu(linear(pool2, 84, p['7.weight'], p['7.bias']));
  const logits = Array.from(linear(hidden, 10, p['9.weight'], p['9.bias']));
  const probabilities = softmax(logits);
  const top3 = probabilities.map((score, index) => ({ index, score }))
    .sort((a, b) => b.score - a.score || a.index - b.index).slice(0, 3);
  const result = { logits, probabilities, top3 };
  if (capture) {
    result.activations = {
      conv1: { data: conv1, shape: [16, 24, 24] },
      relu1: { data: relu1, shape: [16, 24, 24] },
      pool1: { data: pool1, shape: [16, 12, 12] },
      conv2: { data: conv2, shape: [32, 8, 8] },
      relu2: { data: relu2, shape: [32, 8, 8] },
      pool2: { data: pool2, shape: [32, 4, 4] },
    };
  }
  return result;
}
