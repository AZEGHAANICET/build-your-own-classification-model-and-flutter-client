import 'dart:io';
import 'dart:math';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'realtime_detector.dart';
import 'dart:async';
void main() => runApp(const MaterialApp(
  debugShowCheckedModeBanner: false,
  home: LivePage(),
));

class LivePage extends StatefulWidget {
  const LivePage({super.key});
  @override
  State<LivePage> createState() => _LivePageState();
}

class _LivePageState extends State<LivePage> {
  CameraController? _cam;
  final _detector = RealtimeDetector();

  List<Detection> _dets = [];
  String? _error;
  int _fps = 0, _frames = 0;
  late Timer _fpsTimer;

  @override
  void initState() {
    super.initState();
    _init();
    _fpsTimer = Timer.periodic(const Duration(seconds: 1), (_) {
      setState(() {
        _fps = _frames;
        _frames = 0;
      });
    });
  }

  Future<void> _init() async {
    try {
      await _detector.start();
      _detector.onResult = (dets) {
        if (!mounted) return;
        _frames++;
        setState(() => _dets = dets);
      };

      final cameras = await availableCameras();
      final back = cameras.firstWhere(
            (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );

      final isYuv = Platform.isAndroid;
      final controller = CameraController(
        back,
        ResolutionPreset.medium, // plus bas = plus rapide
        enableAudio: false,
        imageFormatGroup:
        isYuv ? ImageFormatGroup.yuv420 : ImageFormatGroup.bgra8888,
      );
      await controller.initialize();

      await controller.startImageStream((image) {
        _detector.process(image, back.sensorOrientation, isYuv);
      });

      if (mounted) setState(() => _cam = controller);
    } catch (e) {
      if (mounted) setState(() => _error = '$e');
    }
  }

  @override
  void dispose() {
    _fpsTimer.cancel();
    _cam?.stopImageStream();
    _cam?.dispose();
    _detector.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final cam = _cam;
    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(
        child: _error != null
            ? Center(
            child: Text(_error!,
                style: const TextStyle(color: Colors.red)))
            : cam == null || !cam.value.isInitialized
            ? const Center(child: CircularProgressIndicator())
            : Center(
          child: AspectRatio(
            // l'aperçu est en portrait : on inverse le ratio paysage
            aspectRatio: 1 / cam.value.aspectRatio,
            child: Stack(fit: StackFit.expand, children: [
              CameraPreview(cam),
              CustomPaint(painter: BoxPainter(_dets)),
              Positioned(
                top: 8,
                left: 8,
                child: Container(
                  padding: const EdgeInsets.symmetric(
                      horizontal: 8, vertical: 4),
                  color: Colors.black54,
                  child: Text('$_fps FPS · ${_dets.length} objet(s)',
                      style: const TextStyle(color: Colors.white)),
                ),
              ),
            ]),
          ),
        ),
      ),
    );
  }
}

class BoxPainter extends CustomPainter {
  final List<Detection> dets;
  BoxPainter(this.dets);

  @override
  void paint(Canvas canvas, Size size) {
    for (final d in dets) {
      final color =
      Colors.primaries[d.label.hashCode.abs() % Colors.primaries.length];
      final rect = Rect.fromLTRB(
        d.box.left * size.width,
        d.box.top * size.height,
        d.box.right * size.width,
        d.box.bottom * size.height,
      );

      canvas.drawRect(
        rect,
        Paint()
          ..color = color
          ..style = PaintingStyle.stroke
          ..strokeWidth = 3,
      );

      final tp = TextPainter(
        text: TextSpan(
          text: ' ${d.label} ${(d.score * 100).toStringAsFixed(0)}% ',
          style: TextStyle(
              color: Colors.white, fontSize: 14, backgroundColor: color),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      tp.paint(canvas, Offset(rect.left, max(0, rect.top - tp.height)));
    }
  }

  @override
  bool shouldRepaint(BoxPainter old) => true;
}