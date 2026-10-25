import { loadModel, runModel } from './inference.mjs';
import { prepareRgba } from './preprocess.mjs';

const $ = (selector) => document.querySelector(selector);
const ui = {
  drop: $('#drop-zone'), input: $('#file-input'), prompt: $('#upload-prompt'), original: $('#original-image'),
  replace: $('#replace-hint'), kind: $('#source-kind'), caption: $('#file-caption'),
  background: $('#background-select'), center: $('#center-checkbox'), processing: $('#processing-note'),
  status: $('#model-status'), empty: $('#result-empty'), content: $('#prediction-content'),
  title: $('#prediction-label'), english: $('#prediction-english'), runtime: $('#runtime-label'),
  scores: $('#score-list'), truth: $('#example-truth'), prepared: $('#prepared-canvas'),
  error: $('#error-message'), reload: $('#reload-button'), examples: $('#examples-list'),
  raw: $('#conv-map'), relu: $('#relu-map'), pool: $('#pool-map'),
  activationStatus: $('#activation-status'),
};

const names = ['Tişört / üst', 'Pantolon', 'Kazak', 'Elbise', 'Kaban', 'Sandalet', 'Gömlek', 'Spor ayakkabı', 'Çanta', 'Bot'];
const shortNames = ['Tişört', 'Pantolon', 'Kazak', 'Elbise', 'Kaban', 'Sandalet', 'Gömlek', 'Spor ayakkabı', 'Çanta', 'Bot'];
const state = { model: null, source: null, revision: 0, objectUrl: null, modelError: false };

function status(kind, message) {
  ui.status.dataset.state = kind;
  ui.status.querySelector('span').textContent = message;
}

function showError(message, { model = false } = {}) {
  clearActivations('Bu görüntü için haritalar hesaplanamadı.');
  ui.error.textContent = message;
  ui.error.hidden = false;
  ui.reload.hidden = !model;
  ui.content.hidden = true;
  ui.empty.hidden = false;
  ui.empty.querySelector('h3').textContent = model ? 'Model şu anda yüklenemedi.' : 'Bu görüntü işlenemedi.';
  ui.empty.querySelector('p').textContent = model ? 'Bağlantını kontrol edip yeniden deneyebilirsin.' : 'Başka bir fotoğraf ya da hazır örnek seçebilirsin.';
  status('error', model ? 'Yükleme hatası' : 'Görüntüyü kontrol et');
}

function begin(message) {
  clearActivations('Yeni görüntü için haritalar hazırlanıyor.');
  ui.error.hidden = true;
  ui.reload.hidden = true;
  ui.content.hidden = true;
  ui.empty.hidden = false;
  ui.empty.querySelector('h3').textContent = message;
  ui.empty.querySelector('p').textContent = 'Görüntü hazırlanıyor ve modelin sınıf skorları hesaplanıyor.';
  status('processing', 'Hesaplanıyor');
}

function imagePixels(image, maxSide = 1024) {
  const scale = Math.min(1, maxSide / Math.max(image.naturalWidth, image.naturalHeight));
  const width = Math.max(1, Math.round(image.naturalWidth * scale));
  const height = Math.max(1, Math.round(image.naturalHeight * scale));
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(image, 0, 0, width, height);
  return { rgba: ctx.getImageData(0, 0, width, height).data, width, height };
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('Görüntü açılamadı. Geçerli bir PNG, JPG veya WebP dosyası seç.'));
    image.src = url;
  });
}

function directExamplePixels(image) {
  if (image.naturalWidth !== 28 || image.naturalHeight !== 28) throw new Error('Hazır örnek 28×28 boyutunda değil.');
  const { rgba } = imagePixels(image, 28);
  const tensor = new Float32Array(784);
  const previewRgba = new Uint8ClampedArray(3136);
  for (let i = 0; i < 784; i++) {
    tensor[i] = rgba[i * 4] / 255;
    previewRgba.set([rgba[i * 4], rgba[i * 4], rgba[i * 4], 255], i * 4);
  }
  return { tensor, previewRgba, warnings: [] };
}

function renderInput(rgba) {
  ui.prepared.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(rgba), 28, 28), 0, 0);
}

function drawActivation(canvas, activation, limit) {
  if (!activation?.data || !activation?.shape) return;
  const [, height, width] = activation.shape;
  canvas.width = width;
  canvas.height = height;
  const pixels = new Uint8ClampedArray(width * height * 4);
  const zero = [229, 232, 224], positive = [200, 110, 84], negative = [83, 119, 172];
  for (let i = 0; i < width * height; i++) {
    const value = activation.data[i];
    const strength = Math.min(1, Math.abs(value) / limit);
    const color = value >= 0 ? positive : negative;
    for (let channel = 0; channel < 3; channel++) pixels[i * 4 + channel] = Math.round(zero[channel] + (color[channel] - zero[channel]) * strength);
    pixels[i * 4 + 3] = 255;
  }
  canvas.getContext('2d').putImageData(new ImageData(pixels, width, height), 0, 0);
}

function clearActivations(message) {
  for (const canvas of [ui.raw, ui.relu, ui.pool]) {
    canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
  }
  ui.activationStatus.textContent = message;
  ui.activationStatus.hidden = false;
}

function renderActivations(activations) {
  if (!activations?.conv1 || !activations?.relu1 || !activations?.pool1) {
    clearActivations('Bu görüntü için haritalar hesaplanamadı.');
    return;
  }
  const raw = activations.conv1;
  const count = raw.shape[1] * raw.shape[2];
  let limit = 1e-8;
  for (let i = 0; i < count; i++) limit = Math.max(limit, Math.abs(raw.data[i]));
  drawActivation(ui.raw, activations.conv1, limit);
  drawActivation(ui.relu, activations.relu1, limit);
  drawActivation(ui.pool, activations.pool1, limit);
  ui.activationStatus.hidden = true;
}

function formatScore(score) {
  const percentage = Math.max(0, Math.min(1, score)) * 100;
  if (score > 0 && percentage < 0.01) return '<%0,01';
  const formatted = percentage.toLocaleString('tr-TR', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  // Preserve the distinction between a rounded display and an exact model output.
  if (score < 1 && formatted === '100,00') return '>%99,99';
  return `%${formatted}`;
}

function renderScores(result, source, elapsed) {
  const top = result.top3 ?? result.probabilities.map((score, index) => ({ index, score })).sort((a, b) => b.score - a.score).slice(0, 3);
  const winner = top[0].index;
  ui.title.textContent = names[winner] ?? state.model.metadata.classes[winner];
  ui.english.textContent = state.model.metadata.classes[winner];
  ui.runtime.textContent = `Tarayıcıda · ${Math.max(1, Math.round(elapsed))} ms`;
  ui.scores.replaceChildren();
  for (const item of top) {
    const row = document.createElement('li');
    row.className = 'score-row';
    const name = document.createElement('span');
    name.textContent = names[item.index] ?? state.model.metadata.classes[item.index];
    const value = document.createElement('strong');
    value.textContent = formatScore(item.score);
    const track = document.createElement('span');
    track.className = 'score-track';
    const fill = document.createElement('span');
    fill.className = 'score-fill';
    fill.style.display = 'block';
    fill.style.width = `${Math.max(0, Math.min(100, item.score * 100))}%`;
    track.append(fill);
    row.append(name, value, track);
    ui.scores.append(row);
  }
  if (source.type === 'example') {
    const correct = winner === source.example.labelIndex;
    ui.truth.hidden = false;
    ui.truth.dataset.correct = String(correct);
    ui.truth.textContent = `Veri seti etiketi: ${names[source.example.labelIndex]} · ${correct ? 'Etiketle eşleşti' : 'Etiketle eşleşmedi'}`;
  } else ui.truth.hidden = true;
  ui.empty.hidden = true;
  ui.content.hidden = false;
  status('ready', 'Tahmin hazır');
}

async function classify(revision) {
  const source = state.source;
  if (!source) return;
  begin('Model görüntüyü inceliyor.');
  try {
    const model = await modelReady;
    if (revision !== state.revision) return;
    const prepared = source.type === 'example'
      ? directExamplePixels(source.image)
      : prepareRgba(source.pixels.rgba, source.pixels.width, source.pixels.height, {
        background: ui.background.value, center: ui.center.checked,
      });
    if (prepared.tensor.length !== 784) throw new Error('Hazırlanan görüntünün boyutu beklenenle eşleşmiyor.');
    renderInput(prepared.previewRgba);
    if (source.type === 'example') {
      ui.processing.textContent = 'Hazır örnek özgün 28×28 pikselleriyle işleniyor. Fotoğraf hazırlama ayarları bu örneğe uygulanmaz.';
    } else {
      const background = prepared.backgroundUsed === 'light' ? 'Açık zemin algısı: parlaklık ters çevrildi.' : 'Koyu zemin algısı: parlaklık yönü korundu.';
      ui.processing.textContent = `${background} ${ui.center.checked ? 'Basit eşikleme ile ürün ortalandı.' : 'Ortalanmadan küçültüldü.'} ${(prepared.warnings ?? []).join(' ')}`;
    }
    // Let loading state paint before the small, synchronous forward pass.
    await new Promise((resolve) => requestAnimationFrame(resolve));
    if (revision !== state.revision) return;
    const start = performance.now();
    const result = runModel(model, prepared.tensor, { capture: true });
    const elapsed = performance.now() - start;
    if (revision !== state.revision) return;
    renderScores(result, source, elapsed);
    renderActivations(result.activations);
  } catch (error) {
    if (revision !== state.revision) return;
    showError(error.message || 'Beklenmeyen bir hata oluştu.', { model: state.modelError });
  }
}

function setSelectedExample(id) {
  ui.examples.querySelectorAll('.example-button').forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.id === String(id))));
}

function displaySource(source, caption) {
  ui.original.src = source.image.src;
  ui.original.alt = source.type === 'example' ? `Fashion-MNIST doğrulama örneği: ${names[source.example.labelIndex]}` : 'Yüklediğin ürün fotoğrafı';
  ui.original.hidden = false;
  ui.prompt.hidden = true;
  ui.replace.hidden = false;
  ui.drop.classList.toggle('dataset-image', source.type === 'example');
  ui.kind.textContent = source.type === 'example' ? 'DOĞRULAMA ÖRNEĞİ' : 'KENDİ FOTOĞRAFIN';
  ui.caption.textContent = caption;
  ui.background.disabled = source.type === 'example';
  ui.center.disabled = source.type === 'example';
}

async function chooseExample(example) {
  const revision = ++state.revision;
  begin('Hazır örnek açılıyor.');
  setSelectedExample(example.id);
  try {
    const image = await loadImage(new URL(example.image, document.baseURI).href);
    if (revision !== state.revision) return;
    if (state.objectUrl) { URL.revokeObjectURL(state.objectUrl); state.objectUrl = null; }
    state.source = { type: 'example', image, example };
    displaySource(state.source, `Fashion-MNIST · kaynak indeks ${example.index}`);
    await classify(revision);
  } catch (error) {
    if (revision === state.revision) showError(error.message);
  }
}

async function chooseFile(file) {
  if (!file) return;
  const revision = ++state.revision;
  setSelectedExample(null);
  if (!['image/png', 'image/jpeg', 'image/webp'].includes(file.type)) {
    showError('PNG, JPG veya WebP biçiminde bir görüntü seç.');
    return;
  }
  if (file.size > 8 * 1024 * 1024) {
    showError('Dosya 8 MB sınırından büyük. Daha küçük bir fotoğraf seç.');
    return;
  }
  begin('Fotoğraf açılıyor.');
  const url = URL.createObjectURL(file);
  try {
    const image = await loadImage(url);
    if (revision !== state.revision) { URL.revokeObjectURL(url); return; }
    if (image.naturalWidth * image.naturalHeight > 36_000_000) throw new Error('Fotoğraf çok büyük. En fazla yaklaşık 36 megapiksel bir görüntü seç.');
    const pixels = imagePixels(image);
    if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
    state.objectUrl = url;
    state.source = { type: 'upload', image, pixels };
    displaySource(state.source, file.name);
    await classify(revision);
  } catch (error) {
    if (state.objectUrl !== url) URL.revokeObjectURL(url);
    if (revision === state.revision) showError(error.message);
  }
}

ui.drop.addEventListener('click', () => ui.input.click());
ui.drop.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); ui.input.click(); }
});
ui.input.addEventListener('change', () => { const file = ui.input.files?.[0]; ui.input.value = ''; chooseFile(file); });
for (const eventName of ['dragenter', 'dragover']) ui.drop.addEventListener(eventName, (event) => { event.preventDefault(); ui.drop.classList.add('dragover'); });
for (const eventName of ['dragleave', 'drop']) ui.drop.addEventListener(eventName, (event) => { event.preventDefault(); ui.drop.classList.remove('dragover'); });
ui.drop.addEventListener('drop', (event) => chooseFile(event.dataTransfer.files?.[0]));
for (const control of [ui.background, ui.center]) control.addEventListener('change', () => { if (state.source?.type === 'upload') classify(++state.revision); });
ui.reload.addEventListener('click', () => location.reload());
document.querySelector('.prepared-section a').addEventListener('click', () => { $('#mechanism-details').open = true; });

const modelReady = loadModel('./model/').then((model) => {
  state.model = model;
  if (!state.source && state.revision === 0) status('ready', 'Model hazır');
  return model;
}).catch((error) => {
  state.modelError = true;
  showError('Model dosyaları yüklenemedi. İnternet bağlantını kontrol edip sayfayı yeniden yükle.', { model: true });
  throw error;
});
// Keep a rejected model promise handled even before a visitor selects an image.
modelReady.catch(() => {});

fetch('./examples/examples.json').then((response) => {
  if (!response.ok) throw new Error('Hazır örnekler yüklenemedi. Kendi fotoğrafınla deneyebilirsin.');
  return response.json();
}).then((examples) => {
  if (!Array.isArray(examples) || !examples.length) throw new Error('Hazır örnek listesi bulunamadı.');
  ui.examples.replaceChildren();
  for (const example of examples) {
    const button = document.createElement('button');
    button.className = 'example-button';
    button.type = 'button';
    button.dataset.id = String(example.id);
    button.setAttribute('aria-pressed', 'false');
    button.setAttribute('aria-label', `${names[example.labelIndex]} doğrulama örneğini çalıştır`);
    const image = document.createElement('img');
    image.src = example.image;
    image.alt = '';
    image.width = 54;
    image.height = 54;
    const label = document.createElement('span');
    label.textContent = shortNames[example.labelIndex];
    button.append(image, label);
    button.addEventListener('click', () => chooseExample(example));
    ui.examples.append(button);
  }
  if (!state.source && state.revision === 0) chooseExample(examples[0]);
}).catch((error) => {
  ui.examples.replaceChildren();
  const note = document.createElement('p');
  note.className = 'examples-loading';
  note.textContent = error.message;
  ui.examples.append(note);
});
