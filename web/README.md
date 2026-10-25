# Browser inference demo

[Live demo](https://gulcan-cnn-kiyafet-laboratuvari.primegull512.chatgpt.site)

The trained 57,190-parameter CNN runs directly in the browser. Uploaded images are processed locally; there is no image inference API. The interface displays the actual 28×28 model input, top-three class scores and intermediate activations.

## Implementation

| File | Responsibility |
|---|---|
| `dist/app.mjs` | File/sample selection, rendering and prediction flow |
| `dist/preprocess.mjs` | Grayscale, transparency, polarity, centering and resizing |
| `dist/inference.mjs` | Convolution, ReLU, average pooling, linear layers and softmax |
| `dist/model/model.json` | Shapes, class order, model identity and benchmark metadata |
| `dist/model/weights.bin` | Exported float32 weights (228,760 bytes) |
| `export_model.py` | Shared Python model export and independent CPU reference generation |

Photo preprocessing converts RGB to grayscale, composites transparency, selects background polarity from edge brightness, optionally crops an intensity-threshold bounding box and preserves aspect ratio when resizing to 28×28. Pixel values are scaled to [0,1]. The preview and tensor use the same grayscale bytes. Dataset samples bypass these heuristics and preserve their original pixels.

The bounding-box method is a brightness heuristic; arbitrary-background segmentation is not implemented. Users can override background polarity and centering. The loader checks weight integrity with SHA256 when WebCrypto is available.

## Local preview

From the repository root:

```powershell
python -m http.server 8000 --bind 127.0.0.1 --directory web/dist
```

Open http://127.0.0.1:8000. No npm dependencies or build step are required.

## Export and checks

With the project installed and dataset downloaded:

```powershell
python web/export_model.py --project . --out runs/web-export/dist --references runs/web-export/references
node web/tests/test-runtime.mjs
node web/tests/test-preprocess.mjs
```

Export reads the frozen checkpoint and official training-source validation examples; it does not train or open the test dataset. The shared factory is `fashion_cnn.models.build_cnn`. Tests do not modify stored references.

Across 100 fixed, class-balanced inputs, JavaScript and independent PyTorch CPU references matched all top1 predictions. Maximum probability/logit/activation errors were approximately 3.43e-7/4.77e-6/6.68e-6 against absolute tolerance 1e-4. This is implementation parity, not 100% classification accuracy. Reference fixtures are not served by the static demo.

## Model limits and attribution

The recorded 91.08% accuracy is from reuse of the official Fashion-MNIST benchmark. Arbitrary-photo accuracy, multi-object detection and unseen-category rejection have not been evaluated. The classifier always chooses among ten classes; scores are not calibrated correctness probabilities. See the [model card](../docs/MODEL_CARD.md).

Source checkpoint SHA256:

`013745dcd66e4d5c462ba64607faed70831ea7e9caa27721afcdc34693b245f2`

Example images derive from Zalando Research [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist). Its MIT notice is included in `dist/examples/LICENSE.txt`.
