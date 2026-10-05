import 'dart:math';
import 'dart:typed_data';
import 'dart:ui' show Rect;
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';

class Detection {
  final Rect box; // coordonnées normalisées (0..1)
  final String label;
  final double score;
  Detection(this.box, this.label, this.score);
}

const int kSize = 640;

/// Resize 640x640, valeurs /255, format CHW (RGB). Exécuté dans un isolate.
Float32List preprocessYolo(Uint8List bytes) {
  final decoded = img.decodeImage(bytes);
  if (decoded == null) throw Exception('Image illisible');
  final resized = img.copyResize(decoded, width: kSize, height: kSize);

  const plane = kSize * kSize;
  final data = Float32List(3 * plane);
  for (int y = 0; y < kSize; y++) {
    for (int x = 0; x < kSize; x++) {
      final p = resized.getPixel(x, y);
      final i = y * kSize + x;
      data[i] = p.r / 255.0;
      data[plane + i] = p.g / 255.0;
      data[2 * plane + i] = p.b / 255.0;
    }
  }
  return data;
}

class YoloDetector {
  late OrtSession _session;
  late List<String> _labels;

  Future<void> load() async {
    OrtEnv.instance.init();
    final d = await rootBundle.load('assets/yolov8n.onnx');
    final bytes = d.buffer.asUint8List(d.offsetInBytes, d.lengthInBytes);
    _session = OrtSession.fromBuffer(bytes, OrtSessionOptions());

    final txt = await rootBundle.loadString('assets/labels.txt');
    _labels = txt.split('\n').where((l) => l.trim().isNotEmpty).toList();
  }

  Future<List<Detection>> detect(
      Uint8List imageBytes, {
        double confThreshold = 0.4,
        double iouThreshold = 0.5,
      }) async {
    final input = await compute(preprocessYolo, imageBytes);

    final tensor =
    OrtValueTensor.createTensorWithDataList(input, [1, 3, kSize, kSize]);
    final runOptions = OrtRunOptions();
    final outputs = _session.run(runOptions, {'images': tensor});

    // out[0] = [84][8400]
    final out = (outputs[0]!.value as List)[0] as List;
    final numBoxes = (out[0] as List).length;
    final numClasses = out.length - 4;

    final candidates = <Detection>[];
    for (int i = 0; i < numBoxes; i++) {
      double best = 0;
      int bestC = -1;
      for (int c = 0; c < numClasses; c++) {
        final s = (out[4 + c][i] as num).toDouble();
        if (s > best) {
          best = s;
          bestC = c;
        }
      }
      if (best < confThreshold) continue;

      final cx = (out[0][i] as num).toDouble();
      final cy = (out[1][i] as num).toDouble();
      final w = (out[2][i] as num).toDouble();
      final h = (out[3][i] as num).toDouble();

      candidates.add(Detection(
        Rect.fromLTRB(
          ((cx - w / 2) / kSize).clamp(0.0, 1.0),
          ((cy - h / 2) / kSize).clamp(0.0, 1.0),
          ((cx + w / 2) / kSize).clamp(0.0, 1.0),
          ((cy + h / 2) / kSize).clamp(0.0, 1.0),
        ),
        _labels[bestC],
        best,
      ));
    }

    tensor.release();
    runOptions.release();
    for (final o in outputs) {
      o?.release();
    }

    return _nms(candidates, iouThreshold);
  }

  /// Non-Maximum Suppression : supprime les boîtes en double d'un même objet.
  List<Detection> _nms(List<Detection> dets, double iouThr) {
    dets.sort((a, b) => b.score.compareTo(a.score));
    final kept = <Detection>[];
    for (final d in dets) {
      final overlap = kept.any(
              (k) => k.label == d.label && _iou(k.box, d.box) > iouThr);
      if (!overlap) kept.add(d);
    }
    return kept;
  }

  double _iou(Rect a, Rect b) {
    final inter = a.intersect(b);
    if (inter.width <= 0 || inter.height <= 0) return 0;
    final interArea = inter.width * inter.height;
    final union = a.width * a.height + b.width * b.height - interArea;
    return interArea / max(union, 1e-9);
  }

  void dispose() {
    _session.release();
    OrtEnv.instance.release();
  }
}