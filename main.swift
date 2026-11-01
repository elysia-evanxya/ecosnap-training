//
// main.swift — TrainEcoSnap
// HANYA jalan di macOS (lewat GitHub Actions runner), TIDAK bisa dijalankan di iPad.
// CreateML framework memang tidak tersedia di iOS/iPadOS SDK.
//
// Cara pakai (otomatis dipanggil oleh .github/workflows/train.yml):
//   swift run --package-path Trainer TrainEcoSnap <path-ke-CleanDataset> <path-output.mlmodel>
//
// Kalau argumen tidak diberikan, pakai default relatif ke working directory saat ini.

import Foundation
import CreateML

let args = CommandLine.arguments
let fm = FileManager.default
let cwd = URL(fileURLWithPath: fm.currentDirectoryPath)

let datasetPath = args.count > 1 ? args[1] : cwd.appendingPathComponent("CleanDataset").path
let outputPath = args.count > 2 ? args[2] : cwd.appendingPathComponent("EcoSnapClassifier.mlmodel").path

let datasetURL = URL(fileURLWithPath: datasetPath)
let outputURL = URL(fileURLWithPath: outputPath)

print("=== EcoSnap Classifier Training ===")
print("Dataset: \(datasetURL.path)")
print("Output model: \(outputURL.path)")

guard fm.fileExists(atPath: datasetURL.path) else {
    print("ERROR: folder dataset tidak ditemukan di \(datasetURL.path)")
    exit(1)
}

// Hitung & tampilkan jumlah foto per kelas sebelum training — supaya kalau ada
// kelas yang kosong/salah path, ketahuan SEBELUM buang waktu training.
if let classFolders = try? fm.contentsOfDirectory(at: datasetURL, includingPropertiesForKeys: nil) {
    print("\n--- Kelas terdeteksi ---")
    for folder in classFolders.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
        guard folder.hasDirectoryPath else { continue }
        let count = (try? fm.contentsOfDirectory(atPath: folder.path).count) ?? 0
        print("\(folder.lastPathComponent): \(count) file")
    }
    print("")
}

do {
    let trainingData = try MLImageClassifier.DataSource.labeledDirectories(at: datasetURL)

    // featureExtractor .scenePrint: transfer learning di atas feature extractor
    // bawaan Apple yang sudah dilatih jutaan gambar — ini yang bikin training
    // cepat (menit, bukan jam) dan tetap akurat walau dataset kita kecil (~988 foto).
    // validationData: nil -> CreateML otomatis sisihkan sebagian data training
    // sebagai validasi (karena dataset kecil, metrik akurasi akan cukup "noisy"
    // — anggap sebagai estimasi kasar, bukan angka presisi).
    let parameters = MLImageClassifier.ModelParameters(
        featureExtractor: .scenePrint(revision: 2),
        validationData: nil,
        maxIterations: 25,
        augmentationOptions: [.rotation, .flip, .blur, .noise, .exposure]
    )

    print("Mulai training (biasanya beberapa menit untuk dataset sekecil ini)...")
    let model = try MLImageClassifier(trainingData: trainingData, parameters: parameters)

    let trainAcc = (1.0 - model.trainingMetrics.classificationError) * 100
    let valAcc = (1.0 - model.validationMetrics.classificationError) * 100

    print("\n=== HASIL TRAINING ===")
    print(String(format: "Training accuracy:   %.1f%%", trainAcc))
    print(String(format: "Validation accuracy: %.1f%%", valAcc))

    let metadata = MLModelMetadata(
        author: "EcoSnap",
        shortDescription: "Klasifikasi jenis sampah (12 kelas) dari foto, untuk app EcoSnap. Dilatih dari dataset internal — bukan dataset publik berskala besar.",
        version: "1.0"
    )

    try model.write(to: outputURL, metadata: metadata)
    print("\nModel tersimpan: \(outputURL.path)")

} catch {
    print("TRAINING GAGAL: \(error)")
    exit(1)
}
