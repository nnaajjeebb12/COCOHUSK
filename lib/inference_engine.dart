import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/services.dart' show rootBundle;
import 'package:tflite_flutter/tflite_flutter.dart';

import 'app_logger.dart';
import 'model_manager.dart';

class PredictionResult {
  final String predictedClass;
  final double confidence;
  final String explanation;

  /// Every class's probability, in the model's output order, the full
  /// breakdown behind [predictedClass], for UI that wants to show "how"
  /// an image was classified rather than just the final answer.
  final Map<String, double> classProbabilities;

  /// The top-scoring class before the rejection layer had a say. Equal to
  /// [predictedClass] when the result wasn't rejected; when it was, this is
  /// what the model's raw output looked closest to.
  final String rawPredictedClass;

  /// Gap between the top and second-highest class probabilities.
  final double margin;

  /// Human-readable reason the rejection layer overrode the raw prediction,
  /// or null if this result wasn't rejected.
  final String? rejectionReason;

  PredictionResult({
    required this.predictedClass,
    required this.confidence,
    required this.explanation,
    this.classProbabilities = const {},
    String? rawPredictedClass,
    this.margin = 0.0,
    this.rejectionReason,
  }) : rawPredictedClass = rawPredictedClass ?? predictedClass;
}

/// Owns the active TFLite interpreter, labels, and rejection config.
/// Shared by still-image and live-camera flows.
class InferenceEngine {
  Interpreter? _interpreter;
  List<String> _labels = const [];
  Map<String, List<double>> _centroids = const {};
  Map<String, Map<String, double>> _distStats = const {};

  ModelSource _source = ModelSource.bundled;
  String? _activePath;
  String? _activeName;
  int _modelVersion = 0;
  List<int>? _inputShape;
  List<int>? _outputShape;
  String? _verificationWarning;

  static const double _confThreshold = 0.70;
  static const double _marginThreshold = 0.20;
  static const double _distanceK = 2.0;
  static const int _inputSize = 224;
  static const int _numChannels = 3;

  bool get isReady => _interpreter != null && _labels.isNotEmpty;

  /// True only when the interpreter loaded AND its tensors were confirmed to
  /// match what the app expects (see [_verifyTensors]). This is stronger
  /// than [isReady], which just means "no exception was thrown".
  bool get isVerified => isReady && _verificationWarning == null;

  /// Human-readable reason [isVerified] is false, or null when there is
  /// nothing to warn about (either not loaded yet, or fully verified).
  String? get verificationWarning => _verificationWarning;

  List<int>? get inputShape => _inputShape;
  List<int>? get outputShape => _outputShape;
  ModelSource get source => _source;
  String? get activePath => _activePath;
  String? get activeName => _activeName;
  int get modelVersion => _modelVersion;
  List<String> get labels => List.unmodifiable(_labels);

  Future<void> loadDefaultModel() async {
    final updateStatus = await ModelManager.checkAndUpdateModel();
    final active = await ModelManager.readActiveModelBytes(
      updateStatus: updateStatus,
      readBundledAsset: () async {
        final data = await rootBundle
            .load('assets/model/coconut_husk_quality_model.tflite');
        return data.buffer.asUint8List(
          data.offsetInBytes,
          data.lengthInBytes,
        );
      },
    );

    final labelsStr = (active.source == ModelSource.local ||
            active.source == ModelSource.cloud)
        ? await ModelManager.readLocalLabels()
        : await rootBundle.loadString('assets/model/class_names.txt');

    _swapInterpreter(Interpreter.fromBuffer(active.bytes));
    _labels = labelsStr
        .split('\n')
        .map((e) => e.trim())
        .where((e) => e.isNotEmpty)
        .toList(growable: false);
    _source = active.source;
    _activePath = active.path;
    _activeName = active.path == null
        ? 'coconut_husk_quality_model.tflite (bundled)'
        : active.path!.split(Platform.pathSeparator).last;
    _modelVersion = await ModelManager.getCurrentModelVersion();

    await _loadRejectionConfig();
    _verifyTensors();

    AppLogger.i('Inference engine: ready, model=$_activeName '
        'source=${_source.name} labels=${_labels.length} '
        'version=$_modelVersion');
  }

  /// Confirms the loaded interpreter's tensors actually match the contract
  /// the app relies on (see the Custom Model Guide): a 4-D input of
  /// [1, 224, 224, 3] and an output whose class count lines up with the
  /// currently loaded labels. Populates [inputShape]/[outputShape] either
  /// way, and sets [verificationWarning] when something looks off so the UI
  /// can surface it instead of silently mispredicting.
  void _verifyTensors() {
    final interpreter = _interpreter;
    if (interpreter == null) {
      _inputShape = null;
      _outputShape = null;
      _verificationWarning = 'No model interpreter loaded.';
      return;
    }

    try {
      _inputShape = interpreter.getInputTensor(0).shape;
      _outputShape = interpreter.getOutputTensor(0).shape;

      final warnings = <String>[];

      final input = _inputShape!;
      if (input.length != 4 ||
          input[1] != _inputSize ||
          input[2] != _inputSize ||
          input[3] != _numChannels) {
        warnings.add('input tensor is $input, expected '
            '[1, $_inputSize, $_inputSize, $_numChannels]');
      }

      final numClasses = _outputShape!.isNotEmpty ? _outputShape!.last : 0;
      if (_labels.isNotEmpty && numClasses != _labels.length) {
        warnings.add('model outputs $numClasses classes but '
            '${_labels.length} labels are loaded, predictions may be '
            'mislabeled');
      }

      _verificationWarning = warnings.isEmpty ? null : warnings.join('; ');
      if (_verificationWarning != null) {
        AppLogger.w('Inference engine: verification warning '
            '$_verificationWarning');
      }
    } catch (e, st) {
      AppLogger.w('Inference engine: tensor verification failed.', e, st);
      _verificationWarning = 'Could not read model tensor shapes.';
    }
  }

  /// Validates the given .tflite bytes by attempting to instantiate an
  /// interpreter. If valid, persists the path and swaps the active model.
  Future<void> loadCustomFromBytes({
    required Uint8List bytes,
    required String displayName,
  }) async {
    AppLogger.i('Inference engine: validating custom model "$displayName" '
        '(${bytes.lengthInBytes} bytes)');

    // Instantiating the interpreter throws if the file isn't a valid TFLite
    // model at all, but a "valid" model can still have the wrong tensor
    // shapes for this app, that check happens below in _verifyTensors(),
    // after we've committed to the swap, so the caller (and UI) can see it.
    final trial = Interpreter.fromBuffer(bytes);

    final imported = await ModelManager.writeCustomModelBytes(
      bytes: bytes,
      fileName: displayName,
    );
    await ModelManager.setCustomModelPath(imported);

    _swapInterpreter(trial);
    _source = ModelSource.custom;
    _activePath = imported;
    _activeName = displayName;
    _verifyTensors();

    if (_verificationWarning != null) {
      AppLogger.w('Inference engine: custom model active - $displayName '
          '(with warning: $_verificationWarning)');
    } else {
      AppLogger.i('Inference engine: custom model active - $displayName '
          '(input=$_inputShape output=$_outputShape)');
    }
  }

  void _swapInterpreter(Interpreter next) {
    try {
      _interpreter?.close();
    } catch (_) {}
    _interpreter = next;
  }

  Future<void> _loadRejectionConfig() async {
    try {
      String? jsonStr = await ModelManager.readLocalRejectionConfig();
      jsonStr ??=
          await rootBundle.loadString('assets/model/rejection_config.json');

      final data = jsonDecode(jsonStr) as Map<String, dynamic>;
      final centroidsRaw =
          (data['centroids'] as Map<String, dynamic>? ?? const {});
      final statsRaw = (data['stats'] as Map<String, dynamic>? ?? const {});

      final centroidsParsed = <String, List<double>>{};
      final statsParsed = <String, Map<String, double>>{};

      centroidsRaw.forEach((cls, vec) {
        if (vec is List) {
          centroidsParsed[cls] = vec
              .map((v) => (v is num) ? v.toDouble() : 0.0)
              .toList(growable: false);
        }
      });
      statsRaw.forEach((cls, st) {
        if (st is Map<String, dynamic>) {
          statsParsed[cls] = {
            'mean_dist': (st['mean_dist'] is num)
                ? (st['mean_dist'] as num).toDouble()
                : 0.0,
            'std_dist': (st['std_dist'] is num)
                ? (st['std_dist'] as num).toDouble()
                : 0.0,
          };
        }
      });

      _centroids = centroidsParsed;
      _distStats = statsParsed;
      AppLogger.d('Rejection config loaded for classes: '
          '${_centroids.keys.toList()}');
    } catch (e, st) {
      AppLogger.w('Rejection config load failed.', e, st);
      _centroids = const {};
      _distStats = const {};
    }
  }

  Future<PredictionResult?> predictFromFile(File file) async {
    final bytes = await file.readAsBytes();
    return predictFromBytes(bytes);
  }

  Future<PredictionResult?> predictFromBytes(Uint8List bytes) async {
    final interpreter = _interpreter;
    if (interpreter == null || _labels.isEmpty) return null;

    try {
      final normalized = await _preprocessImage(bytes, _inputSize, _inputSize);

      final inputShape = interpreter.getInputTensor(0).shape;
      final expected = inputShape[1] * inputShape[2] * inputShape[3];
      if (normalized.length != expected) {
        AppLogger.w('Preprocessed length ${normalized.length} != '
            'expected $expected (model input $inputShape)');
        return null;
      }

      final input = Float32List.fromList(normalized).reshape(inputShape);
      final outputShape = interpreter.getOutputTensor(0).shape;
      final outLen = outputShape.reduce((a, b) => a * b);
      final output = List<double>.filled(outLen, 0.0).reshape(outputShape);

      interpreter.run(input, output);

      final probs = List<double>.from(output[0] as List);
      if (probs.isEmpty) return null;

      double sumProbs = 0.0;
      for (final p in probs) {
        sumProbs += p;
      }
      if (sumProbs > 0.0 && (sumProbs < 0.99 || sumProbs > 1.01)) {
        for (int i = 0; i < probs.length; i++) {
          probs[i] = probs[i] / sumProbs;
        }
      }

      int maxIndex = 0;
      double maxScore = probs[0];
      for (int i = 1; i < probs.length; i++) {
        if (probs[i] > maxScore) {
          maxScore = probs[i];
          maxIndex = i;
        }
      }

      double top2 = 0.0;
      for (int i = 0; i < probs.length; i++) {
        if (i == maxIndex) continue;
        if (probs[i] > top2) top2 = probs[i];
      }
      final margin = maxScore - top2;

      String predictedClass =
          (maxIndex < _labels.length) ? _labels[maxIndex] : 'unknown';
      final confidence = maxScore;

      final classProbabilities = <String, double>{
        for (int i = 0; i < _labels.length && i < probs.length; i++)
          _labels[i]: probs[i],
      };

      bool rejected = false;
      String? rejectionReason;
      if (confidence < _confThreshold) {
        rejected = true;
        rejectionReason = 'Confidence '
            '(${(confidence * 100).toStringAsFixed(1)}%) is below the '
            '${(_confThreshold * 100).toStringAsFixed(0)}% minimum needed '
            'to accept a prediction.';
      } else if (probs.length >= 2 && margin < _marginThreshold) {
        rejected = true;
        rejectionReason = 'The top two classes were too close '
            '(${(margin * 100).toStringAsFixed(1)}% apart, '
            '${(_marginThreshold * 100).toStringAsFixed(0)}% minimum needed) '
            'for the model to confidently tell them apart.';
      } else if (_centroids.isNotEmpty &&
          _distStats.isNotEmpty &&
          _labels.length == probs.length) {
        bool allFar = true;
        for (int i = 0; i < _labels.length; i++) {
          final centroid = _centroids[_labels[i]];
          final stats = _distStats[_labels[i]];
          if (centroid == null || stats == null) continue;
          final dist = _euclidean(probs, centroid);
          final mean = stats['mean_dist'] ?? 0.0;
          final std = stats['std_dist'] ?? 0.0;
          if (dist <= mean + _distanceK * std) {
            allFar = false;
            break;
          }
        }
        if (allFar) {
          rejected = true;
          rejectionReason = 'The prediction pattern for this image does '
              'not closely resemble any known class from prior examples, '
              'it likely is not a coconut husk, or the photo is unclear.';
        }
      }

      final finalClass = rejected ? 'rejected' : predictedClass;
      return PredictionResult(
        predictedClass: finalClass,
        confidence: confidence,
        explanation: _explanationForClass(finalClass),
        classProbabilities: classProbabilities,
        rawPredictedClass: predictedClass,
        margin: margin,
        rejectionReason: rejectionReason,
      );
    } catch (e, st) {
      AppLogger.e('Inference error', e, st);
      return null;
    }
  }

  double _euclidean(List<double> a, List<double> b) {
    if (a.length != b.length) return double.infinity;
    double sum = 0.0;
    for (int i = 0; i < a.length; i++) {
      final d = a[i] - b[i];
      sum += d * d;
    }
    return sum <= 0 ? 0 : math.sqrt(sum);
  }

  Future<List<double>> _preprocessImage(
      Uint8List bytes, int width, int height) async {
    final codec = await ui.instantiateImageCodec(bytes);
    final frame = await codec.getNextFrame();
    final src = frame.image;

    final recorder = ui.PictureRecorder();
    final canvas = ui.Canvas(recorder);
    canvas.drawImageRect(
      src,
      ui.Rect.fromLTWH(0, 0, src.width.toDouble(), src.height.toDouble()),
      ui.Rect.fromLTWH(0, 0, width.toDouble(), height.toDouble()),
      ui.Paint(),
    );
    final resized = await recorder.endRecording().toImage(width, height);
    final byteData =
        await resized.toByteData(format: ui.ImageByteFormat.rawRgba);
    if (byteData == null) {
      throw Exception('Failed to convert image to byte data');
    }

    final rgba = byteData.buffer.asUint8List();
    final n = width * height;
    final out = List<double>.filled(n * _numChannels, 0.0);
    for (int i = 0; i < n; i++) {
      final ri = i * 4;
      final wi = i * 3;
      // Raw 0-255 float32 values, NOT normalized to 0-1 here. Both shipped
      // models bake their own normalization into the graph as its first
      // layer (an explicit Rescaling(1/255) in the small custom CNN; an
      // internal Rescaling+Normalization inside EfficientNetB0 itself,
      // which is why keras.applications.efficientnet.preprocess_input is a
      // no-op). Dividing by 255 again here double-normalizes the input,
      // which saturates the network into a single constant prediction
      // regardless of image content, confirmed by feeding solid-color and
      // noise images and seeing 100% "overmature" for all of them.
      out[wi] = rgba[ri].toDouble();
      out[wi + 1] = rgba[ri + 1].toDouble();
      out[wi + 2] = rgba[ri + 2].toDouble();
    }
    src.dispose();
    resized.dispose();
    return out;
  }

  String _explanationForClass(String predictedClass) {
    switch (predictedClass) {
      case 'immature':
        return 'Immature: the husk has not fully developed yet, fibers are '
            'underdeveloped and moisture content is high. Best left to '
            'mature further before processing.';
      case 'mature':
        return 'Mature: the ideal harvest stage, fibers are fully '
            'developed with optimal moisture content, top processing '
            'quality.';
      case 'overmature':
        return 'Overmature: the husk has aged past its ideal stage, '
            'fibers may be dry or degraded, reducing processing quality.';
      case 'rejected':
        return 'Rejected: the image does not sufficiently match any known '
            'coconut husk maturity stage. Please capture a clear coconut '
            'husk image on a plain background.';
      default:
        return 'Unknown maturity stage.';
    }
  }

  void dispose() {
    try {
      _interpreter?.close();
    } catch (_) {}
    _interpreter = null;
  }
}
