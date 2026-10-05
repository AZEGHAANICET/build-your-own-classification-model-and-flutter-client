import 'dart:async';
import 'dart:isolate';
import 'dart:math';
import 'dart:typed_data';
import 'dart:ui' show Rect;
import 'package:camera/camera.dart';
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';

const int kSize = 640;

class Detection {
  final Rect box; // normalisé 0..1, repère image redressée
  final String label;
  final double score;
  Detection(this.box, this.label, this.score);
}

/// Gère l'isolate qui exécute le modèle.
class RealtimeDetector {
  late SendPort _toIsolate;
  late List<String> _labels;
  Isolate? _isolate;
  final _receive = ReceivePort();
  bool busy = false;
  void Function(List<Detection>)? onResult;

  Future<void> start() async {
    final model = await rootBundle.load('assets/yolov8n.onnx');
    final modelBytes =
    model.buffer.asUint8List(model.offsetInBytes, model.lengthInBytes);
    final txt = await rootBundle.loadString('assets/labels.txt');
    _labels = txt.split('\n').where((l) => l.trim().isNotEmpty).toList();

    final ready = Completer<void>();
    _receive.listen((msg) {
      if (msg is SendPort) {
        _toIsolate = msg;
        ready.complete();
      } else if (msg is List<double>) {
        busy = false;
        onResult?.call(_decode(msg));
      } else {
        busy = false; // erreur côté isolate
      }
    });
    _isolate = await Isolate.spawn(_entry, [_receive.sendPort, modelBytes]);
    await ready.future;
  }

  List<Detection> _decode(List<double> f) {
    final res = <Detection>[];
    for (int i = 0; i + 5 < f.length; i += 6) {
      res.add(Detection(
        Rect.fromLTRB(f[i], f[i + 1], f[i + 2], f[i + 3]),
        _labels[f[i + 5].toInt()],
        f[i + 4],
      ));
    }
    return res;
  }

  /// Envoie une image caméra (ignorée si l'isolate est occupé).
  void process(CameraImage img, int rotation, bool isYuv) {
    if (busy) return;
    busy = true;
    _toIsolate.send({
      'w': img.width,
      'h': img.height,
      'rot': rotation,
      'yuv': isYuv,
      'p0': img.planes[0].bytes,
      'rs0': img.planes[0].bytesPerRow,
      if (isYuv) ...{
        'p1': img.planes[1].bytes,
        'p2': img.planes[2].bytes,
        'rs1': img.planes[1].bytesPerRow,
        'ps1': img.planes[1].bytesPerPixel ?? 1,
      },
    });
  }

  void dispose() {
    _isolate?.kill(priority: Isolate.immediate);
    _receive.close();
  }
}

// ---------------------------------------------------------------------------
// Code exécuté DANS l'isolate
// ---------------------------------------------------------------------------

void _entry(List args) {
  final SendPort toMain = args[0];
  final Uint8List modelBytes = args[1];

  OrtEnv.instance.init();
  final opts = OrtSessionOptions()..setIntraOpNumThreads(4);
  final session = OrtSession.fromBuffer(modelBytes, opts);

  final port = ReceivePort();
  toMain.send(port.sendPort);

  port.listen((msg) {
    try {
      final input = _preprocess(msg as Map);
      final tensor =
      OrtValueTensor.createTensorWithDataList(input, [1, 3, kSize, kSize]);
      final run = OrtRunOptions();
      final outputs = session.run(run, {'images': tensor});
      final result = _postprocess(outputs[0]!.value as List);
      tensor.release();
      run.release();
      for (final o in outputs) {
        o?.release();
      }
      toMain.send(result);
    } catch (e) {
      toMain.send('error: $e');
    }
  });
}

Float32List _preprocess(Map m) {
  final int w = m['w'], h = m['h'], rot = m['rot'];
  final bool yuv = m['yuv'];
  final Uint8List p0 = m['p0'];
  final int rs0 = m['rs0'];
  final Uint8List? p1 = m['p1'], p2 = m['p2'];
  final int rs1 = m['rs1'] ?? 0, ps1 = m['ps1'] ?? 1;

  // Dimensions de l'image une fois redressée
  final upW = (rot % 180 == 0) ? w : h;
  final upH = (rot % 180 == 0) ? h : w;

  const plane = kSize * kSize;
  final data = Float32List(3 * plane);

  for (int oy = 0; oy < kSize; oy++) {
    final uy = (oy * upH) ~/ kSize;
    for (int ox = 0; ox < kSize; ox++) {
      final ux = (ox * upW) ~/ kSize;

      int sx, sy;
      switch (rot) {
        case 90:
          sx = uy;
          sy = h - 1 - ux;
          break;
        case 180:
          sx = w - 1 - ux;
          sy = h - 1 - uy;
          break;
        case 270:
          sx = w - 1 - uy;
          sy = ux;
          break;
        default:
          sx = ux;
          sy = uy;
      }

      double r, g, b;
      if (yuv) {
        final yv = p0[sy * rs0 + sx];
        final uvIdx = (sy >> 1) * rs1 + (sx >> 1) * ps1;
        final u = p1![uvIdx] - 128;
        final v = p2![uvIdx] - 128;
        r = yv + 1.402 * v;
        g = yv - 0.344136 * u - 0.714136 * v;
        b = yv + 1.772 * u;
      } else {
        final idx = sy * rs0 + sx * 4; // BGRA
        b = p0[idx].toDouble();
        g = p0[idx + 1].toDouble();
        r = p0[idx + 2].toDouble();
      }

      final i = oy * kSize + ox;
      data[i] = r.clamp(0, 255) / 255.0;
      data[plane + i] = g.clamp(0, 255) / 255.0;
      data[2 * plane + i] = b.clamp(0, 255) / 255.0;
    }
  }
  return data;
}

/// Retourne une liste plate : [l, t, r, b, score, classe, ...]
List<double> _postprocess(List raw,
    {double conf = 0.4, double iouThr = 0.5}) {
  final out = raw[0] as List; // [84][8400]
  final numBoxes = (out[0] as List).length;
  final numClasses = out.length - 4;

  final cands = <List<double>>[]; // [l,t,r,b,score,cls]
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
    if (best < conf) continue;
    final cx = (out[0][i] as num).toDouble();
    final cy = (out[1][i] as num).toDouble();
    final w = (out[2][i] as num).toDouble();
    final h = (out[3][i] as num).toDouble();
    cands.add([
      ((cx - w / 2) / kSize).clamp(0.0, 1.0),
      ((cy - h / 2) / kSize).clamp(0.0, 1.0),
      ((cx + w / 2) / kSize).clamp(0.0, 1.0),
      ((cy + h / 2) / kSize).clamp(0.0, 1.0),
      best,
      bestC.toDouble(),
    ]);
  }

  // NMS par classe
  cands.sort((a, b) => b[4].compareTo(a[4]));
  final kept = <List<double>>[];
  for (final d in cands) {
    final dup = kept.any((k) => k[5] == d[5] && _iou(k, d) > iouThr);
    if (!dup) kept.add(d);
  }
  return [for (final k in kept) ...k];
}

double _iou(List<double> a, List<double> b) {
  final l = max(a[0], b[0]), t = max(a[1], b[1]);
  final r = min(a[2], b[2]), btm = min(a[3], b[3]);
  final iw = r - l, ih = btm - t;
  if (iw <= 0 || ih <= 0) return 0;
  final inter = iw * ih;
  final areaA = (a[2] - a[0]) * (a[3] - a[1]);
  final areaB = (b[2] - b[0]) * (b[3] - b[1]);
  return inter / max(areaA + areaB - inter, 1e-9);
}