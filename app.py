import os, io, base64, threading, json
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify
from PIL import Image
import timm
from torchvision import transforms
from huggingface_hub import hf_hub_download

app = Flask(__name__)

MODEL_ID = "Uzzyy/dermiq-v2-classifier"
model = None
config = None
transform = None
model_ready = False

def load_model_background():
    global model, config, transform, model_ready
    print("Loading DermIQ V2 model...")

    config_path = hf_hub_download(repo_id=MODEL_ID, filename="config.json")
    with open(config_path) as f:
        config = json.load(f)

    model_path = hf_hub_download(repo_id=MODEL_ID, filename="model.pth")
    m = timm.create_model('tf_efficientnetv2_s', pretrained=False, num_classes=config['num_classes'])
    m.load_state_dict(torch.load(model_path, map_location='cpu'))
    m.eval()
    model = m

    transform = transforms.Compose([
        transforms.Resize((config['img_size'], config['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=config['mean'], std=config['std']),
    ])

    model_ready = True
    print("Model ready!")

threading.Thread(target=load_model_background, daemon=True).start()

def add_cors(response, status=200):
    r = app.make_response((response, status))
    r.headers["Access-Control-Allow-Origin"] = "*"
    r.headers["Access-Control-Allow-Headers"] = "Content-Type"
    r.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    r.headers["Content-Type"] = "application/json"
    return r

def decode_image(b64: str) -> Image.Image:
    for p in ["data:image/jpeg;base64,","data:image/png;base64,",
              "data:image/webp;base64,","data:image/heic;base64,"]:
        b64 = b64.replace(p, "")
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

def is_skin(image_pil: Image.Image, threshold=0.15):
    img = image_pil.resize((128, 128)).convert("RGB")
    arr = np.array(img, dtype=np.float32)
    R, G, B = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    cmax = np.maximum(np.maximum(R, G), B)
    cmin = np.minimum(np.minimum(R, G), B)
    kovac = (R>95)&(G>40)&(B>20)&((cmax-cmin)>15)&(np.abs(R-G)>15)&(R>G)&(R>B)
    Y  =  0.299*R + 0.587*G + 0.114*B
    Cb = -0.169*R - 0.331*G + 0.500*B + 128
    Cr =  0.500*R - 0.419*G - 0.081*B + 128
    ycbcr = (Y>80)&(Cb>=77)&(Cb<=127)&(Cr>=133)&(Cr<=173)
    ratio = float((kovac|ycbcr).sum()) / (128*128)
    return ratio >= threshold, round(ratio, 3)

def compute_entropy(predictions):
    probs = np.array([p["score"] for p in predictions], dtype=np.float64)
    probs = np.clip(probs, 1e-10, 1.0)
    return float(-np.sum(probs * np.log(probs)) / np.log(len(probs)))

@app.route("/")
def health():
    return add_cors(json.dumps({"status": "running", "model": MODEL_ID, "model_ready": model_ready, "version": "v2"}))

@app.route("/classify", methods=["POST", "OPTIONS"])
def classify():
    if request.method == "OPTIONS":
        return add_cors(json.dumps({}))

    if not model_ready:
        return add_cors(json.dumps({
            "error": "Model warming up. Please try again in 20 seconds.",
            "code": "NOT_READY"
        }), 503)

    data = request.json
    if not data or "image" not in data:
        return add_cors(json.dumps({"error": "No image provided."}), 400)

    try:
        image_pil = decode_image(data["image"])
    except Exception:
        return add_cors(json.dumps({"error": "Could not read image."}), 400)

    skin_ok, skin_ratio = is_skin(image_pil)
    if not skin_ok:
        return add_cors(json.dumps({
            "error": "No skin detected. Please upload a close-up photo of a skin lesion.",
            "code": "NO_SKIN", "skin_ratio": skin_ratio,
        }), 422)

    input_tensor = transform(image_pil).unsqueeze(0)
    with torch.no_grad():
        probs = F.softmax(model(input_tensor), dim=1)[0]

    predictions = [
        {"label": config['id2label'][str(i)], "score": round(float(p), 6)}
        for i, p in enumerate(probs.tolist())
    ]
    predictions.sort(key=lambda x: x["score"], reverse=True)

    top_conf = predictions[0]["score"]
    conf_gap = predictions[0]["score"] - predictions[1]["score"]
    entropy = compute_entropy(predictions)

    if top_conf < 0.25 or entropy > 0.88 or conf_gap < 0.08:
        return add_cors(json.dumps({
            "error": "No clear skin lesion detected. Please take a close-up photo with the lesion centred.",
            "code": "NO_LESION",
        }), 422)

    return add_cors(json.dumps({
        "predictions": predictions,
        "gradcam": None,
        "top_class": predictions[0]["label"],
        "skin_ratio": skin_ratio,
        "version": "v2",
    }))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))