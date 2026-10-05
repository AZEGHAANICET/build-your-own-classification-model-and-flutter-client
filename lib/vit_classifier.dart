import "dart:math";
import "dart:typed_data";
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';


class Prediction {
  final String label;
  final double score;
  Prediction(this.label, this.score);
}

Float32List preprocessImage(Uint8List bytes) {
  final decoded = img.decodeImage(bytes);
  if (decoded == null) throw Exception('Image illisible');

  final resized = img.copyResize(decoded, width: 224, height: 224);
  const plane = 224 * 224;
  final data = Float32List(3 * plane);

  for (int y = 0; y < 224; y++) {
    for (int x = 0; x < 224; x++) {
      final p = resized.getPixel(x, y);
      final i = y * 224 + x;
      data[i] = (p.r / 255.0 - 0.5) / 0.5;
      data[plane + i] = (p.g / 255.0 - 0.5) / 0.5;
      data[2 * plane + i] = (p.b / 255.0 - 0.5) / 0.5;
    }
  }
  return data;
}


class ViTClassifier {
  late OrtSession _session;
  late List<String> _labels;

  Future<void> load() async {
    OrtEnv.instance.init();

    final modelData = await rootBundle.load('assets/vit.onnx');
    final modelBytes = modelData.buffer
        .asUint8List(modelData.offsetInBytes, modelData.lengthInBytes);

    final options = OrtSessionOptions();
    _session = OrtSession.fromBuffer(modelBytes, options);

    final labelsText = await rootBundle.loadString('assets/labels.txt');
    _labels = labelsText.split('\n').where((l) => l.trim().isNotEmpty).toList();
  }

  Future<List<Prediction>> predict(Uint8List imageBytes, {int topK = 5}) async {
    final input = await compute(preprocessImage, imageBytes);

    final inputTensor =
    OrtValueTensor.createTensorWithDataList(input, [1, 3, 224, 224]);
    final runOptions = OrtRunOptions();

    final outputs = _session.run(runOptions, {'pixel_values': inputTensor});
    final logits =
    List<double>.from((outputs[0]!.value as List<List<double>>)[0]);

    inputTensor.release();
    runOptions.release();
    for (final o in outputs) {
      o?.release();
    }

    // Softmax
    final maxV = logits.reduce(max);
    final exps = logits.map((v) => exp(v - maxV)).toList();
    final sum = exps.reduce((a, b) => a + b);
    final probs = exps.map((v) => v / sum).toList();

    final idx = List<int>.generate(probs.length, (i) => i)
      ..sort((a, b) => probs[b].compareTo(probs[a]));

    return idx.take(topK).map((i) => Prediction(_labels[i], probs[i])).toList();
  }

  void dispose() {
    _session.release();
    OrtEnv.instance.release();
  }
}