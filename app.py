"""
SmartCoop ML API — Pipeline 3 Model
Classifier → Feces Model / Body Model
"""
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import onnxruntime as ort

app = FastAPI(title="SmartCoop ML API — 3 Model Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading models...")
sess_classifier = ort.InferenceSession("models/classifier.onnx")
sess_feces      = ort.InferenceSession("models/feces_model.onnx")
sess_body       = ort.InferenceSession("models/body_model.onnx")
print("All 3 models loaded.")

CLASSIFIER_CLASSES    = ["Feces", "Physique"]
FECES_DISEASE_CLASSES = ["Coccidiosis", "Healthy", "New Castle Disease", "Salmonella"]
BODY_DISEASE_CLASSES  = [
    "Bumblefoot", "CRD", "Coryza", "Fowlpox", "Healthy",
    "eye abnormality", "eye swelling", "normal-eyes",
    "normal-posture", "paralyzed", "wing droop",
]
IMG_SIZE = (224, 224)


def preprocess(image_bytes: bytes) -> np.ndarray:
    """Bytes → float32 tensor [1, 3, H, W]  (0–1, tanpa normalisasi)"""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    return np.expand_dims(arr, axis=0)


def run_onnx(session: ort.InferenceSession, tensor: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    outputs    = session.run(None, {input_name: tensor})[0]
    logits     = outputs[0]
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


def get_recommendation(disease: str) -> str:
    recs = {
        "Coccidiosis":        "Berikan anticoccidial (Amprolium). Jaga kebersihan litter kandang.",
        "New Castle Disease": "SEGERA isolasi kandang. Hubungi dokter hewan. Vaksinasi darurat.",
        "Salmonella":         "Isolasi ayam bergejala. Berikan antibiotik sesuai resep. Tingkatkan sanitasi.",
        "Bumblefoot":         "Bersihkan dan balut luka kaki. Konsultasikan antibiotik topikal.",
        "CRD":                "Berikan tylosin atau enrofloxacin. Perbaiki ventilasi kandang.",
        "Coryza":             "Berikan sulfonamide atau eritromisin. Isolasi ayam sakit.",
        "Fowlpox":            "Tidak ada obat spesifik. Vaksinasi ayam yang belum terjangkit.",
        "eye abnormality":    "Periksa lebih lanjut oleh dokter hewan.",
        "eye swelling":       "Kemungkinan CRD/Coryza. Isolasi dan konsultasikan dokter hewan.",
        "normal-eyes":        "Kondisi mata normal. Tidak ada tindakan diperlukan.",
        "normal-posture":     "Postur normal. Tidak ada tindakan diperlukan.",
        "paralyzed":          "Kemungkinan Marek's Disease. Pisahkan segera.",
        "wing droop":         "Gejala Newcastle atau cedera. Isolasi dan periksa.",
        "Healthy":            "Tidak ada tindakan. Lanjutkan pemantauan rutin.",
    }
    return recs.get(disease, "Konsultasikan dengan dokter hewan.")


@app.get("/")
def root():
    return {"status": "SmartCoop ML API is running",
            "pipeline": "3-model (classifier → feces_model / body_model)",
            "version": "1.0.0"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": {"classifier": "loaded", "feces_model": "loaded", "body_model": "loaded"},
        "classes": {
            "classifier":  CLASSIFIER_CLASSES,
            "feces_model": FECES_DISEASE_CLASSES,
            "body_model":  BODY_DISEASE_CLASSES,
        }
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    # ── Validasi: coba buka dengan PIL, bukan cek content-type ──────────────
    # (content-type dari Flutter tidak selalu reliable)
    try:
        test_img = Image.open(io.BytesIO(image_bytes))
        test_img.verify()  # raise jika bukan gambar valid
    except Exception:
        raise HTTPException(
            status_code=422,
            detail={
                "error":   "File tidak valid",
                "message": "File tidak bisa dibaca sebagai gambar. "
                           "Pastikan mengirim file foto yang valid (jpg/png).",
                "classifier_confidence": 0,
            }
        )

    tensor = preprocess(image_bytes)

    # ── STEP 1: Classifier ──────────────────────────────────────────────────
    clf_probs = run_onnx(sess_classifier, tensor)
    clf_idx   = int(np.argmax(clf_probs))
    clf_label = CLASSIFIER_CLASSES[clf_idx]
    clf_conf  = float(clf_probs[clf_idx]) * 100

    if clf_conf < 40.0:
        raise HTTPException(
            status_code=422,
            detail={
                "error":                  "Foto tidak dikenali",
                "message":                "Pastikan foto menampilkan feses ayam "
                                          "atau fisik ayam dengan jelas.",
                "classifier_confidence":  round(clf_conf, 2),
            }
        )

    # ── STEP 2: Disease detection ───────────────────────────────────────────
    if clf_label == "Feces":
        disease_probs   = run_onnx(sess_feces, tensor)
        disease_classes = FECES_DISEASE_CLASSES
        photo_type      = "Feces"
        model_used      = "feces_model.onnx"
    else:
        disease_probs   = run_onnx(sess_body, tensor)
        disease_classes = BODY_DISEASE_CLASSES
        photo_type      = "Physique"
        model_used      = "body_model.onnx"

    top_idx        = int(np.argmax(disease_probs))
    top_disease    = disease_classes[top_idx]
    top_confidence = float(disease_probs[top_idx]) * 100

    all_classes = {
        disease_classes[i]: round(float(disease_probs[i]) * 100, 2)
        for i in range(len(disease_classes))
    }

    return {
        "disease":            top_disease,
        "confidence":         round(top_confidence, 2),
        "needs_verification": top_confidence < 60.0,
        "recommendation":     get_recommendation(top_disease),
        "all_classes":        all_classes,
        "pipeline": {
            "photo_type":            photo_type,
            "classifier_label":      clf_label,
            "classifier_confidence": round(clf_conf, 2),
            "model_used":            model_used,
        }
    }
