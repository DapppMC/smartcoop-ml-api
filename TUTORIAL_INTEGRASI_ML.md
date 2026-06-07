# 🤖 SmartCoop — Tutorial Integrasi 3 Model ML (.onnx)

---

## Arsitektur Pipeline

```
[Flutter kirim foto]
        │
        ▼
  Model 1: classifier.onnx
  "Ini foto tai atau fisik ayam?"
        │
        ├── "feces" ──► Model 2: feces_model.onnx
        │                "Penyakit apa dari foto tai?"
        │
        └── "body"  ──► Model 3: body_model.onnx
                         "Penyakit apa dari foto fisik ayam?"
        │
        ▼
  JSON response → Flutter tampilkan hasil
        │
        ▼
  Simpan ke Supabase detection_logs
```

**Kenapa .onnx bukan .pth?**
- Docker image lebih kecil (onnxruntime 10MB vs PyTorch 800MB+)
- Inferensi lebih cepat
- Tidak perlu install PyTorch di server

---

## LANGKAH 1 — Konversi .pth → .onnx (di komputer teman kamu)

Kalau model masih dalam format `.pth`, teman kamu perlu konversi dulu.
Jalankan script ini di environment tempat model dilatih:

```python
# convert_to_onnx.py — jalankan ini di komputer teman kamu
import torch

# ── Ganti dengan class model kamu ──────────────────────────────
from your_model_file import YourModelClass   # sesuaikan ini

IMG_SIZE = 224   # sesuaikan dengan ukuran input model

def convert(pth_path, onnx_path, num_classes):
    model = YourModelClass(num_classes=num_classes)
    model.load_state_dict(torch.load(pth_path, map_location='cpu'))
    model.eval()

    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    print(f"Saved: {onnx_path}")

# Konversi ketiga model
convert("classifier.pth",  "models/classifier.onnx",  num_classes=2)
convert("feces_model.pth",  "models/feces_model.onnx", num_classes=5)  # sesuaikan jumlah kelas
convert("body_model.pth",   "models/body_model.onnx",  num_classes=5)  # sesuaikan jumlah kelas
```

Kalau model sudah .onnx → langsung ke Langkah 2.

---

## LANGKAH 2 — Struktur Folder Project

```
smartcoop-ml-api/
├── app.py              ← server FastAPI
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── models/             ← ⚠️ BUAT FOLDER INI
    ├── classifier.onnx
    ├── feces_model.onnx
    └── body_model.onnx
```

Copy ketiga file `.onnx` ke folder `models/`.

---

## LANGKAH 3 — Sesuaikan Label Kelas di app.py

Buka `app.py`, cari bagian ini dan sesuaikan dengan label training model kamu:

```python
# Model 1 — label harus persis 2: nama untuk "tai" dan nama untuk "fisik ayam"
CLASSIFIER_CLASSES = ["feces", "body"]   # ⚠️ sesuaikan urutan dengan training

# Model 2 — label penyakit dari foto tai
FECES_DISEASE_CLASSES = [
    "Healthy",
    "Newcastle Disease",
    "Salmonellosis",
    "Avian Influenza",
    "Infectious Bronchitis",
]   # ⚠️ urutan HARUS sama persis dengan urutan saat training

# Model 3 — label penyakit dari foto fisik ayam
BODY_DISEASE_CLASSES = [
    "Healthy",
    "Newcastle Disease",
    "Marek's Disease",
    "Avian Influenza",
    "Infectious Bronchitis",
]   # ⚠️ sesuaikan
```

> **Cara cek urutan label yang benar:**
> Tanya teman yang train model — minta dia print `class_to_idx` dari dataset-nya:
> ```python
> print(train_dataset.class_to_idx)
> # Output: {'Healthy': 0, 'Newcastle Disease': 1, ...}
> # Urutan angka indeks = urutan di list CLASS_NAMES
> ```

---

## LANGKAH 4 — Sesuaikan Preprocessing (Penting!)

Di `app.py` fungsi `preprocess()`, ada normalisasi ImageNet:
```python
mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])
```

Ini standar untuk model yang pakai pretrained (ResNet, EfficientNet, dll).
Kalau teman kamu pakai normalisasi berbeda saat training, sesuaikan di sini.

Tanya teman: "transforms.Normalize kamu pakai mean dan std berapa?"

---

## LANGKAH 5 — Test Lokal dengan Docker

```bash
# Masuk ke folder
cd smartcoop-ml-api

# Build image
docker build -t smartcoop-ml .

# Jalankan (port 8000)
docker run -p 8000:8000 smartcoop-ml
```

Cek server berjalan:
```bash
curl http://localhost:8000/health
```
Respons yang diharapkan:
```json
{
  "status": "ok",
  "models": {
    "classifier": "loaded",
    "feces_model": "loaded",
    "body_model": "loaded"
  }
}
```

Test prediksi dengan foto:
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@foto_tai_ayam.jpg"
```

Contoh respons sukses:
```json
{
  "disease": "Salmonellosis",
  "confidence": 87.34,
  "needs_verification": false,
  "recommendation": "Isolasi ayam bergejala...",
  "all_classes": {
    "Healthy": 5.12,
    "Salmonellosis": 87.34,
    ...
  },
  "pipeline": {
    "photo_type": "feces",
    "classifier_label": "feces",
    "classifier_confidence": 94.21,
    "model_used": "feces_model.onnx"
  }
}
```

---

## LANGKAH 6 — Deploy ke Railway

```bash
# Push ke GitHub dulu
git init
git add .
git commit -m "SmartCoop ML API - 3 model pipeline"
git remote add origin https://github.com/USERNAME/smartcoop-ml-api.git
git push -u origin main
```

1. Buka [railway.app](https://railway.app) → login GitHub
2. **New Project** → **Deploy from GitHub repo** → pilih repo
3. Railway otomatis detect Dockerfile dan build
4. Tunggu ~5-10 menit
5. **Settings** → **Networking** → **Generate Domain**
6. Catat URL: `https://smartcoop-ml-api-XXXX.up.railway.app`

---

## LANGKAH 7 — Update Flutter ml_service.dart

Buka `lib/features/detection/services/ml_service.dart`,
**ganti SELURUH isi file** dengan kode berikut:

```dart
import 'dart:io';
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:flutter_riverpod/flutter_riverpod.dart';

final mlServiceProvider = Provider<MlService>((ref) => MlServiceRemote());

class MlDetectionResult {
  final String disease;
  final double confidence;
  final String? recommendation;
  final bool needsVerification;
  final Map<String, double> allClasses;
  final String photoType;          // "feces" atau "body"
  final double classifierConfidence;

  const MlDetectionResult({
    required this.disease,
    required this.confidence,
    this.recommendation,
    required this.needsVerification,
    required this.allClasses,
    required this.photoType,
    required this.classifierConfidence,
  });
}

class MlRecoveryScore {
  final double score;
  final String trend;
  final String? note;
  const MlRecoveryScore({required this.score, required this.trend, this.note});
}

abstract class MlService {
  Future<MlDetectionResult> analyzeImage(File imageFile);
  Future<MlRecoveryScore> calculateRecoveryScore({
    required List<double?> weightHistory,
    required List<String> latestSymptoms,
    required String behavior,
    required String feedIntake,
  });
}

class MlServiceRemote implements MlService {
  // ⚠️ GANTI dengan URL Railway kamu
  static const String _baseUrl =
      'https://smartcoop-ml-api-XXXX.up.railway.app';

  static const Duration _timeout = Duration(seconds: 30);

  @override
  Future<MlDetectionResult> analyzeImage(File imageFile) async {
    try {
      final request = http.MultipartRequest(
        'POST',
        Uri.parse('$_baseUrl/predict'),
      );
      request.files.add(
        await http.MultipartFile.fromPath('file', imageFile.path),
      );

      final streamed = await request.send().timeout(_timeout);
      final response = await http.Response.fromStream(streamed);

      if (response.statusCode == 422) {
        // Foto tidak dikenali sebagai tai/fisik ayam
        final body = jsonDecode(response.body) as Map<String, dynamic>;
        final detail = body['detail'] as Map<String, dynamic>? ?? {};
        throw Exception(
            detail['message'] ?? 'Foto tidak dikenali. Coba foto ulang.');
      }

      if (response.statusCode != 200) {
        throw Exception('Server error ${response.statusCode}');
      }

      final json = jsonDecode(response.body) as Map<String, dynamic>;
      final pipeline = json['pipeline'] as Map<String, dynamic>? ?? {};

      return MlDetectionResult(
        disease: json['disease'] as String,
        confidence: (json['confidence'] as num).toDouble(),
        recommendation: json['recommendation'] as String?,
        needsVerification: json['needs_verification'] as bool? ?? false,
        allClasses: (json['all_classes'] as Map<String, dynamic>).map(
          (k, v) => MapEntry(k, (v as num).toDouble()),
        ),
        photoType: pipeline['photo_type'] as String? ?? 'unknown',
        classifierConfidence:
            (pipeline['classifier_confidence'] as num?)?.toDouble() ?? 0.0,
      );
    } on SocketException {
      throw Exception('Tidak bisa terhubung ke server ML. Cek koneksi internet.');
    } catch (e) {
      rethrow;
    }
  }

  @override
  Future<MlRecoveryScore> calculateRecoveryScore({
    required List<double?> weightHistory,
    required List<String> latestSymptoms,
    required String behavior,
    required String feedIntake,
  }) async {
    double score = 50.0;
    if (weightHistory.length >= 2) {
      final last = weightHistory.last;
      final prev = weightHistory[weightHistory.length - 2];
      if (last != null && prev != null) {
        if (last > prev) score += 20;
        if (last < prev * 0.95) score -= 20;
      }
    }
    if (behavior == 'active') score += 15;
    if (behavior == 'very_weak') score -= 20;
    if (feedIntake == 'normal') score += 10;
    if (feedIntake == 'none') score -= 15;
    if (latestSymptoms.isEmpty) score += 10;
    score = score.clamp(0, 100);
    return MlRecoveryScore(
      score: score,
      trend: score >= 70 ? 'improving' : score < 40 ? 'declining' : 'stable',
    );
  }
}
```

---

## LANGKAH 8 — Tambah package http di pubspec.yaml

```yaml
dependencies:
  http: ^1.2.1   # ← tambahkan ini
```

```bash
flutter pub get
```

---

## LANGKAH 9 — Update Tampilan Hasil di detection_screen.dart

Di `lib/features/detection/screens/detection_screen.dart`,
cari widget `_MlResultCard` dan tambahkan info photo_type.

Cari baris ini di dalam `_MlResultCard.build()`:
```dart
InfoRow(label: 'Terdeteksi', value: result.disease),
```

Tambahkan satu baris DI ATAS-nya:
```dart
InfoRow(
  label: 'Jenis Foto',
  value: result.photoType == 'feces' ? '💩 Feses Ayam' : '🐔 Fisik Ayam',
),
InfoRow(
  label: 'Model Dipakai',
  value: result.photoType == 'feces' ? 'Feces Disease Model' : 'Body Disease Model',
),
```

---

## Checklist Akhir

- [ ] Konversi .pth → .onnx (kalau belum)
- [ ] Taruh 3 file .onnx di folder `models/`
- [ ] Sesuaikan `CLASS_NAMES` di app.py
- [ ] Sesuaikan normalisasi preprocessing jika berbeda dari ImageNet
- [ ] `docker build` → `docker run` → test lokal
- [ ] Push ke GitHub → deploy Railway
- [ ] Ganti `_baseUrl` di ml_service.dart
- [ ] `flutter pub get` → `flutter run`
- [ ] Test ambil foto dari HP → hasil muncul

---

## Troubleshooting Umum

| Error | Penyebab | Solusi |
|-------|----------|--------|
| `422 Foto tidak dikenali` | Foto bukan tai/ayam atau terlalu blur | Ambil foto lebih dekat dan jelas |
| `Output shape mismatch` | Jumlah kelas di CLASS_NAMES salah | Cek jumlah kelas saat training |
| `Timeout` | Model terlalu berat / server sleep | Pakai Railway (bukan Render) atau upgrade |
| `Wrong prediction` | Normalisasi berbeda dari training | Sesuaikan mean/std di preprocess() |
