import 'dart:async';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import 'app_logger.dart';
import 'inference_engine.dart';

/// Full-screen camera preview that runs prediction on a stream of stills
/// roughly every [_predictIntervalMs] ms. The user can tap **Capture** to
/// lock in the current frame and return its [PredictionResult] to the caller.
class LiveCameraPage extends StatefulWidget {
  final InferenceEngine engine;

  const LiveCameraPage({super.key, required this.engine});

  @override
  State<LiveCameraPage> createState() => _LiveCameraPageState();
}

class _LiveCameraPageState extends State<LiveCameraPage>
    with WidgetsBindingObserver {
  CameraController? _controller;
  List<CameraDescription> _cameras = const [];
  int _cameraIndex = 0;

  bool _initializing = true;
  String? _error;

  Timer? _tickTimer;
  bool _inferring = false;
  bool _capturing = false;
  bool _frozen = false;

  PredictionResult? _liveResult;

  // How often to grab a frame for live inference. 1500 ms keeps the device
  // responsive on mid-range Android phones; lower it on a flagship if needed.
  static const int _predictIntervalMs = 1500;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _bootstrap();
  }

  Future<void> _bootstrap() async {
    try {
      final cams = await availableCameras();
      if (cams.isEmpty) {
        setState(() {
          _initializing = false;
          _error = 'No cameras detected on this device.';
        });
        return;
      }
      _cameras = cams;
      // Prefer the rear-facing camera when present.
      final rearIndex = cams.indexWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
      );
      _cameraIndex = rearIndex >= 0 ? rearIndex : 0;
      await _startCamera(_cameras[_cameraIndex]);
    } catch (e, st) {
      AppLogger.e('Live camera bootstrap failed.', e, st);
      if (!mounted) return;
      setState(() {
        _initializing = false;
        _error = _humanizeCameraError(e);
      });
    }
  }

  Future<void> _startCamera(CameraDescription description) async {
    final controller = CameraController(
      description,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.jpeg,
    );
    await controller.initialize();
    if (!mounted) {
      await controller.dispose();
      return;
    }
    _controller = controller;
    setState(() => _initializing = false);
    _startTicker();
    AppLogger.i('Live camera started: ${description.name} '
        '(${description.lensDirection.name})');
  }

  void _startTicker() {
    _tickTimer?.cancel();
    _tickTimer = Timer.periodic(
      const Duration(milliseconds: _predictIntervalMs),
      (_) => _captureAndInfer(),
    );
  }

  Future<void> _captureAndInfer() async {
    if (_inferring || _capturing || _frozen) return;
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) return;
    if (!widget.engine.isReady) return;

    _inferring = true;
    try {
      final XFile shot = await controller.takePicture();
      final bytes = await shot.readAsBytes();
      final result = await widget.engine.predictFromBytes(bytes);
      try {
        await File(shot.path).delete();
      } catch (_) {}
      if (!mounted) return;
      if (result != null) {
        setState(() => _liveResult = result);
      }
    } catch (e, st) {
      AppLogger.w('Live inference tick failed.', e, st);
    } finally {
      _inferring = false;
    }
  }

  Future<void> _captureFinal() async {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) return;
    if (_capturing) return;

    setState(() {
      _capturing = true;
      _frozen = true;
    });
    _tickTimer?.cancel();

    try {
      final XFile shot = await controller.takePicture();
      final bytes = await shot.readAsBytes();
      final result = await widget.engine.predictFromBytes(bytes);

      try {
        await File(shot.path).delete();
      } catch (_) {}

      if (!mounted) return;
      if (result != null) {
        // Persist the shot to app storage so the home page can display it.
        Navigator.of(context).pop<CapturedFrameResult>(
          CapturedFrameResult(bytes: bytes, prediction: result),
        );
      } else {
        setState(() {
          _capturing = false;
          _frozen = false;
        });
        _startTicker();
        if (!mounted) return;
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
              content: Text('Could not analyze the captured frame.')),
        );
      }
    } catch (e, st) {
      AppLogger.e('Final capture failed.', e, st);
      if (!mounted) return;
      setState(() {
        _capturing = false;
        _frozen = false;
      });
      _startTicker();
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Capture failed: ${_humanizeCameraError(e)}')),
      );
    }
  }

  Future<void> _switchCamera() async {
    if (_cameras.length < 2) return;
    _tickTimer?.cancel();
    final old = _controller;
    setState(() {
      _controller = null;
      _initializing = true;
    });
    try {
      await old?.dispose();
    } catch (_) {}
    _cameraIndex = (_cameraIndex + 1) % _cameras.length;
    await _startCamera(_cameras[_cameraIndex]);
  }

  String _humanizeCameraError(Object e) {
    if (e is CameraException) {
      switch (e.code) {
        case 'CameraAccessDenied':
        case 'CameraAccessDeniedWithoutPrompt':
          return 'Camera permission denied. Enable it in Settings → Apps → HuskTech → Permissions.';
        case 'CameraAccessRestricted':
          return 'Camera access is restricted on this device.';
        case 'AudioAccessDenied':
          return 'Microphone permission denied (audio is disabled; this should not happen).';
      }
      return 'Camera error: ${e.code}: ${e.description ?? ""}';
    }
    return e.toString();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) return;
    if (state == AppLifecycleState.inactive ||
        state == AppLifecycleState.paused) {
      _tickTimer?.cancel();
      controller.dispose();
      _controller = null;
    } else if (state == AppLifecycleState.resumed) {
      if (_cameras.isNotEmpty) {
        _startCamera(_cameras[_cameraIndex]);
      }
    }
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _tickTimer?.cancel();
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(child: _buildBody(context)),
    );
  }

  Widget _buildBody(BuildContext context) {
    if (_error != null) {
      return Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.no_photography, color: Colors.white70, size: 64),
            const SizedBox(height: 16),
            Text(_error!,
                textAlign: TextAlign.center,
                style: const TextStyle(color: Colors.white)),
            const SizedBox(height: 24),
            FilledButton.icon(
              onPressed: () => Navigator.of(context).pop(),
              icon: const Icon(Icons.arrow_back),
              label: const Text('Back'),
            ),
          ],
        ),
      );
    }

    if (_initializing ||
        _controller == null ||
        !_controller!.value.isInitialized) {
      return const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            CircularProgressIndicator(),
            SizedBox(height: 12),
            Text('Starting camera...', style: TextStyle(color: Colors.white70)),
          ],
        ),
      );
    }

    final preview = CameraPreview(_controller!);

    return Stack(
      fit: StackFit.expand,
      children: [
        Center(child: preview),
        Positioned(
          top: 8,
          left: 8,
          child: _circleIconButton(
            icon: Icons.arrow_back,
            onPressed: () => Navigator.of(context).pop(),
          ),
        ),
        if (_cameras.length > 1)
          Positioned(
            top: 8,
            right: 8,
            child: _circleIconButton(
              icon: Icons.cameraswitch,
              onPressed: _capturing ? null : _switchCamera,
            ),
          ),
        Align(
          alignment: Alignment.bottomCenter,
          child: _buildOverlay(context),
        ),
      ],
    );
  }

  Widget _circleIconButton(
      {required IconData icon, required VoidCallback? onPressed}) {
    return Material(
      color: Colors.black54,
      shape: const CircleBorder(),
      child: IconButton(
        icon: Icon(icon, color: Colors.white),
        onPressed: onPressed,
      ),
    );
  }

  Widget _buildOverlay(BuildContext context) {
    final r = _liveResult;
    String headline = 'Aim at a coconut husk...';
    String sub = 'Live prediction will update every ~$_predictIntervalMs ms.';
    Color badgeColor = Colors.grey;

    if (r != null) {
      headline = '${_formatClass(r.predictedClass)} '
          '(${(r.confidence * 100).toStringAsFixed(1)}%)';
      sub = r.explanation;
      badgeColor = _colorFor(r.predictedClass);
    }

    return Container(
      margin: const EdgeInsets.all(16),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.65),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 12,
                height: 12,
                decoration:
                    BoxDecoration(color: badgeColor, shape: BoxShape.circle),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  headline,
                  style: const TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.w600,
                    fontSize: 16,
                  ),
                ),
              ),
              if (_inferring && r != null)
                const SizedBox(
                  width: 16,
                  height: 16,
                  child: CircularProgressIndicator(
                    strokeWidth: 2,
                    color: Colors.white70,
                  ),
                ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            sub,
            style: const TextStyle(color: Colors.white70, fontSize: 12),
          ),
          const SizedBox(height: 12),
          SizedBox(
            width: double.infinity,
            child: FilledButton.icon(
              onPressed: _capturing ? null : _captureFinal,
              icon: _capturing
                  ? const SizedBox(
                      width: 16,
                      height: 16,
                      child: CircularProgressIndicator(
                          strokeWidth: 2, color: Colors.white),
                    )
                  : const Icon(Icons.camera),
              label: Text(_capturing ? 'Capturing...' : 'Capture'),
            ),
          ),
        ],
      ),
    );
  }

  Color _colorFor(String predictedClass) {
    switch (predictedClass) {
      case 'immature':
        return Colors.orange;
      case 'mature':
        return Colors.green;
      case 'overmature':
        return Colors.brown;
      case 'rejected':
        return Colors.red;
      default:
        return Colors.blueGrey;
    }
  }

  String _formatClass(String c) {
    if (c.isEmpty) return c;
    return c
        .split('_')
        .map((p) =>
            p.isEmpty ? p : p[0].toUpperCase() + p.substring(1).toLowerCase())
        .join(' ');
  }
}

class CapturedFrameResult {
  final List<int> bytes;
  final PredictionResult prediction;

  CapturedFrameResult({required this.bytes, required this.prediction});
}
