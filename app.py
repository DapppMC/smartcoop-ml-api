"""
SmartCoop ML API — Pipeline 3 Model
-------------------------------------
Model 1 (classifier.onnx)  : Feces atau Physique?
Model 2 (feces_model.onnx) : deteksi penyakit dari foto feses
Model 3 (body_model.onnx)  : deteksi penyakit dari foto fisik ayam
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

# ── Load ketiga model saat server start ─────────────────────────────────────
print("Loading models...")
sess_classifier = ort.InferenceSession("models/classifier.onnx")
sess_feces      = ort.InferenceSession("models/feces_model.onnx")
sess_body       = ort.InferenceSession("models/body_model.onnx")
print("All 3 models loaded successfully.")

# ── Label Kelas (sesuai confusion matrix) ────────────────────────────────────

# Model 1: classifier — urutan alfabetis: Feces, Physique
CLASSIFIER_CLASSES = ["Feces", "Physique"]

# Model 2: feces_model — urutan alfabetis: Coccidiosis, Healthy, New Castle Disease, Salmonella
FECES_DISEASE_CLASSES = [
    "Coccidiosis",
    "Healthy",
    "New Castle Disease",
    "Salmonella",
]

# Model 3: body_model — urutan alfabetis sesuai confusion matrix
BODY_DISEASE_CLASSES = [
    "Bumblefoot",
    "CRD",
    "Coryza",
    "Fowlpox",
    "Healthy",
    "eye abnormality",
    "eye swelling",
    "normal-eyes",
    "normal-posture",
    "paralyzed",
    "wing droop",
]

# Ukuran input model (sesuaikan jika berbeda)
IMG_SIZE = (224, 224)


# ── Helper Functions ─────────────────────────────────────────────────────────

def preprocess(image_bytes: bytes, size: tuple = IMG_SIZE) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(size)
    arr = np.array(img, dtype=np.float32) / 255.0  # ToTensor() → nilai 0.0-1.0
    arr = arr.transpose(2, 0, 1)                    # HWC → CHW
    return np.expand_dims(arr, axis=0)              # → [1, 3, 224, 224]


def run_onnx(session: ort.InferenceSession, tensor: np.ndarray) -> np.ndarray:
    """Jalankan inferensi ONNX → kembalikan probabilitas tiap kelas (softmax)"""
    input_name = session.get_inputs()[0].name
    outputs    = session.run(None, {input_name: tensor})[0]
    logits     = outputs[0]
    # Softmax manual
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


def get_recommendation(disease: str) -> str:
    recs = {
        # ── Hasil dari feces_model ──────────────────────────────────────────
        "Coccidiosis":        "Berikan anticoccidial (Amprolium). Jaga kebersihan "
                              "litter kandang. Hindari kelembaban berlebih.",
        "New Castle Disease": "SEGERA isolasi kandang. Hubungi dokter hewan. "
                              "Vaksinasi darurat pada ayam yang belum terjangkit.",
        "Salmonella":         "Isolasi ayam bergejala. Berikan antibiotik sesuai "
                              "resep dokter. Tingkatkan sanitasi air dan pakan.",
        # ── Hasil dari body_model ───────────────────────────────────────────
        "Bumblefoot":         "Bersihkan dan balut luka di telapak kaki. "
                              "Konsultasikan pemberian antibiotik topikal ke dokter hewan.",
        "CRD":                "Berikan tylosin atau enrofloxacin. "
                              "Perbaiki ventilasi kandang dan kurangi kepadatan.",
        "Coryza":             "Berikan sulfonamide atau eritromisin. "
                              "Isolasi ayam sakit dari populasi sehat.",
        "Fowlpox":            "Tidak ada pengobatan spesifik. Lakukan vaksinasi "
                              "pada ayam yang belum terjangkit di kandang lain.",
        "eye abnormality":    "Kondisi mata abnormal terdeteksi. Periksa lebih "
                              "lanjut oleh dokter hewan untuk diagnosis pasti.",
        "eye swelling":       "Kemungkinan gejala CRD atau Coryza. "
                              "Isolasi ayam dan konsultasikan ke dokter hewan segera.",
        "normal-eyes":        "Kondisi mata normal. Tidak ada tindakan diperlukan.",
        "normal-posture":     "Postur tubuh normal. Tidak ada tindakan diperlukan.",
        "paralyzed":          "Kemungkinan Marek's Disease. Pisahkan segera — "
                              "tidak ada pengobatan. Vaksinasi flock yang belum terjangkit.",
        "wing droop":         "Sayap turun bisa gejala Newcastle atau cedera fisik. "
                              "Isolasi ayam dan periksa lebih lanjut.",
        # ── Healthy (ada di kedua model) ────────────────────────────────────
        "Healthy":            "Tidak ada tindakan diperlukan. Lanjutkan pemantauan rutin.",
    }
    return recs.get(disease, "Konsultasikan dengan dokter hewan untuk penanganan lebih lanjut.")


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status":   "SmartCoop ML API is running",
        "pipeline": "3-model (classifier → feces_model / body_model)",
        "version":  "1.0.0",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": {
            "classifier":  "loaded",
            "feces_model": "loaded",
            "body_model":  "loaded",
        },
        "classes": {
            "classifier":  CLASSIFIER_CLASSES,
            "feces_model": FECES_DISEASE_CLASSES,
            "body_model":  BODY_DISEASE_CLASSES,
        }
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Pipeline 3 model:
      1. classifier.onnx  → tentukan apakah foto Feces atau Physique
      2a. Jika Feces    → feces_model.onnx → deteksi penyakit dari feses
      2b. Jika Physique → body_model.onnx  → deteksi penyakit dari fisik ayam
    """
    # Validasi tipe file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar (jpg/png)")

    image_bytes = await file.read()
    tensor      = preprocess(image_bytes)

    # ── STEP 1: Classifier — Feces atau Physique? ────────────────────────────
    clf_probs = run_onnx(sess_classifier, tensor)
    clf_idx   = int(np.argmax(clf_probs))
    clf_label = CLASSIFIER_CLASSES[clf_idx]   # "Feces" atau "Physique"
    clf_conf  = float(clf_probs[clf_idx]) * 100

    # Tolak foto jika confidence classifier terlalu rendah
    if clf_conf < 40.0:
        raise HTTPException(
            status_code=422,
            detail={
                "error":                  "Foto tidak dikenali",
                "message":                "Pastikan foto menampilkan feses ayam "
                                          "atau fisik ayam dengan jelas. "
                                          "Hindari foto buram atau terlalu gelap.",
                "classifier_confidence":  round(clf_conf, 2),
            }
        )

    # ── STEP 2: Deteksi penyakit ─────────────────────────────────────────────
    if clf_label == "Feces":
        disease_probs   = run_onnx(sess_feces, tensor)
        disease_classes = FECES_DISEASE_CLASSES
        photo_type      = "Feces"
        model_used      = "feces_model.onnx"
    else:  # "Physique"
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
        # ── Hasil utama yang dipakai Flutter ─────────────────────────────────
        "disease":            top_disease,
        "confidence":         round(top_confidence, 2),
        "needs_verification": top_confidence < 60.0,
        "recommendation":     get_recommendation(top_disease),
        "all_classes":        all_classes,

        # ── Info pipeline (untuk debug & logging ke Supabase) ─────────────
        "pipeline": {
            "photo_type":            photo_type,
            "classifier_label":      clf_label,
            "classifier_confidence": round(clf_conf, 2),
            "model_used":            model_used,
        }
    }