import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/foundation.dart' show visibleForTesting;
import 'package:flutter/services.dart' show rootBundle;
import 'package:tflite_flutter/tflite_flutter.dart';

import 'app_logger.dart';
import 'husk_gate.dart';
import 'model_manager.dart';

class PredictionResult {
  final String predictedClass;
  final double confidence;
  final String explanation;

  /// Short "typically used for" note for the predicted class, shown as a
  /// discreet secondary line alongside [explanation]. Null for classes with
  /// no meaningful use-case list (e.g. "rejected").
  final String? useCases;

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

  /// Image-level husk-likeness measurements behind the "is this even a
  /// coconut husk?" check, or null for results produced before that check
  /// existed (old history rows) or when the image could not be measured.
  final HuskFeatures? huskFeatures;

  /// The thresholds this particular result was judged against, captured at
  /// prediction time. Stored per result rather than read live off the
  /// engine so that a history entry always shows the reference values it
  /// was actually decided by, not whatever the strictness slider happens to
  /// be set to now.
  final double confThresholdUsed;
  final double marginThresholdUsed;
  final double huskThresholdUsed;

  /// Which check rejected this result, or [RejectionCheck.none] when it was
  /// accepted. Lets the UI point at the row that failed instead of making
  /// the user match prose against numbers.
  final RejectionCheck rejectedBy;

  PredictionResult({
    required this.predictedClass,
    required this.confidence,
    required this.explanation,
    this.useCases,
    this.classProbabilities = const {},
    String? rawPredictedClass,
    this.margin = 0.0,
    this.rejectionReason,
    this.huskFeatures,
    this.confThresholdUsed = 0.0,
    this.marginThresholdUsed = 0.0,
    this.huskThresholdUsed = 0.0,
    this.rejectedBy = RejectionCheck.none,
  }) : rawPredictedClass = rawPredictedClass ?? predictedClass;

  /// True when this result carries the threshold values it was judged
  /// against. False for history rows written before they were recorded, so
  /// the UI can omit the reference rather than show zeroes.
  bool get hasThresholdReference => confThresholdUsed > 0.0;
}

/// The checks a prediction has to pass, in the order they are applied.
enum RejectionCheck { none, notHusk, confidence, margin, pattern }

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

  // The confidence/margin thresholds below are user-tunable via a single
  // 0.0 (lenient) - 1.0 (strict) "strictness" slider (see setStrictness).
  // Strictness is in-memory only, not persisted; it resets to 0.5 (Balanced)
  // on every app launch, same idea as the desktop client tool's
  // conf_threshold/margin_threshold it mirrors.
  //
  // These bounds were originally 0.50-0.90 confidence / 0.10-0.30 margin
  // (chosen to reproduce the app's old fixed 0.70/0.20 defaults at the
  // midpoint) but that range turned out to be the wrong shape for this
  // model: measured across all 180 real training photos, confidence was
  // 96-100% (median 100%) and margin was 92.1-100% (median 100%) - meaning
  // the old range sat entirely below where the model actually operates, so
  // the slider had no visible effect on any real photo at any position.
  // Recalibrated so the Strict end actually rejects the model's real
  // low-confidence tail instead of never triggering.
  static const double _lenientConfThreshold = 0.85;
  static const double _strictConfThreshold = 0.999;
  static const double _lenientMarginThreshold = 0.80;
  static const double _strictMarginThreshold = 0.99;

  // Minimum husk-likeness score (see HuskGate) an image has to reach before
  // the model's answer is trusted at all. Measured over the project's 1332
  // husk photos against 22 non-husk images: real husks scored 0.829 and up
  // (median 1.000), every non-husk image scored 0.239 and below. Both ends
  // of this range therefore sit in the empty gap between the two - at the
  // strict end it rejected 0 of 1332 real husks while still catching all 22
  // non-husk images, so moving the slider trades margin for margin rather
  // than trading real rejections for real misses.
  static const double _lenientHuskThreshold = 0.25;
  static const double _strictHuskThreshold = 0.65;
  static const double _defaultStrictness = 0.5;

  double _strictness = _defaultStrictness;
  double _confThreshold =
      _lerp(_lenientConfThreshold, _strictConfThreshold, _defaultStrictness);
  double _marginThreshold = _lerp(
      _lenientMarginThreshold, _strictMarginThreshold, _defaultStrictness);
  double _huskThreshold =
      _lerp(_lenientHuskThreshold, _strictHuskThreshold, _defaultStrictness);

  static const double _distanceK = 2.0;
  static const int _inputSize = 224;
  static const int _numChannels = 3;

  static double _lerp(double a, double b, double t) => a + (b - a) * t;

  /// Current strictness, 0.0 (lenient) - 1.0 (strict).
  double get strictness => _strictness;
  double get confThreshold => _confThreshold;
  double get marginThreshold => _marginThreshold;
  double get huskThreshold => _huskThreshold;

  /// The confidence/margin/husk thresholds at a given strictness, without
  /// changing the engine's own setting. Used by the Information page to
  /// print the full reference table of what each preset means.
  static ({double confidence, double margin, double husk}) thresholdsAt(
      double strictness) {
    final t = strictness.clamp(0.0, 1.0);
    return (
      confidence: _lerp(_lenientConfThreshold, _strictConfThreshold, t),
      margin: _lerp(_lenientMarginThreshold, _strictMarginThreshold, t),
      husk: _lerp(_lenientHuskThreshold, _strictHuskThreshold, t),
    );
  }

  /// Adjusts how strict the rejection layer is for future predictions.
  /// Does not affect predictions already made (e.g. history entries keep
  /// whatever verdict they were given at the time).
  void setStrictness(double value) {
    _strictness = value.clamp(0.0, 1.0);
    _confThreshold =
        _lerp(_lenientConfThreshold, _strictConfThreshold, _strictness);
    _marginThreshold =
        _lerp(_lenientMarginThreshold, _strictMarginThreshold, _strictness);
    _huskThreshold =
        _lerp(_lenientHuskThreshold, _strictHuskThreshold, _strictness);
  }

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

  /// [onProgress], when given, is called with a human-readable phase (and a
  /// 0.0-1.0 fraction during the one step that can actually report real
  /// progress: downloading a new cloud model) so the UI can show something
  /// better than a static "Loading model..." message during what can be a
  /// multi-minute cloud download.
  Future<void> loadDefaultModel({ModelLoadProgress? onProgress}) async {
    final updateStatus =
        await ModelManager.checkAndUpdateModel(onProgress: onProgress);
    onProgress?.call('Reading model...');
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

    // Whatever readActiveModelBytes resolved (custom/local/cloud/bundled)
    // can still turn out to be unusable — a corrupted cache file, or a
    // "download" that actually returned an HTML error page (see
    // ModelManager's HTML-sniff check, which catches most of that case
    // before it even gets here, but not everything: a truncated download,
    // disk corruption, etc). Rather than leave the engine with no working
    // model at all in that case, fall back to the bundled asset, which is
    // the one source guaranteed to actually be there.
    try {
      final labelsStr = (active.source == ModelSource.local ||
              active.source == ModelSource.cloud)
          ? await ModelManager.readLocalLabels()
          : await rootBundle.loadString('assets/model/class_names.txt');

      onProgress?.call('Loading model into memory...');
      final interpreter = Interpreter.fromBuffer(active.bytes);

      _swapInterpreter(interpreter);
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

      onProgress?.call('Verifying model...');
      await _loadRejectionConfig();
      _verifyTensors();

      AppLogger.i('Inference engine: ready, model=$_activeName '
          'source=${_source.name} labels=${_labels.length} '
          'version=$_modelVersion');
    } catch (e, st) {
      if (active.source == ModelSource.bundled) {
        // Already the bundled asset and it still failed — nothing left to
        // fall back to; let the caller's own error handling take over.
        rethrow;
      }
      AppLogger.e(
        'Inference engine: failed to load ${active.source.name} model '
        '(${active.path ?? "?"}); falling back to the bundled asset.',
        e,
        st,
      );
      await loadBundledModel(onProgress: onProgress);
    }
  }

  /// Forces the bundled asset (model + labels + rejection config, all three
  /// from the app package, none from cache) to become active right now,
  /// bypassing the cloud check and any locally cached model entirely -
  /// distinct from [loadDefaultModel], which prefers cloud/cached-cloud and
  /// only falls back to bundled when nothing else is available. Also clears
  /// any custom override, so it doesn't immediately reassert itself.
  ///
  /// Does not delete the locally cached cloud files, so a later
  /// [loadDefaultModel] call (e.g. the ordinary reload/relaunch path) can
  /// still find and use them again - this is a one-time "use bundled right
  /// now" action, not a permanent opt-out of cloud updates.
  Future<void> loadBundledModel({ModelLoadProgress? onProgress}) async {
    onProgress?.call('Loading bundled model...');
    await ModelManager.clearCustomModelPath();

    final data =
        await rootBundle.load('assets/model/coconut_husk_quality_model.tflite');
    final bytes =
        data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
    final labelsStr =
        await rootBundle.loadString('assets/model/class_names.txt');

    onProgress?.call('Loading model into memory...');
    _swapInterpreter(Interpreter.fromBuffer(bytes));
    _labels = labelsStr
        .split('\n')
        .map((e) => e.trim())
        .where((e) => e.isNotEmpty)
        .toList(growable: false);
    _source = ModelSource.bundled;
    _activePath = null;
    _activeName = 'coconut_husk_quality_model.tflite (bundled)';
    _modelVersion = 0;

    onProgress?.call('Verifying model...');
    await _loadBundledRejectionConfig();
    _verifyTensors();

    AppLogger.i('Inference engine: forced to bundled asset - '
        'labels=${_labels.length}');
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

  /// Local cache preferred, falling back to the bundled asset. Used by the
  /// normal (cloud-preferred) load path.
  Future<void> _loadRejectionConfig() async {
    String? jsonStr = await ModelManager.readLocalRejectionConfig();
    jsonStr ??=
        await rootBundle.loadString('assets/model/rejection_config.json');
    await _applyRejectionConfigJson(jsonStr);
  }

  /// Always the bundled asset, ignoring any locally cached rejection config
  /// even if one exists. Used by [loadBundledModel], so "use the bundled
  /// model" means the whole bundled trio (model + labels + rejection
  /// config), not a mix of bundled model with a stale cached config.
  Future<void> _loadBundledRejectionConfig() async {
    final jsonStr =
        await rootBundle.loadString('assets/model/rejection_config.json');
    await _applyRejectionConfigJson(jsonStr);
  }

  Future<void> _applyRejectionConfigJson(String jsonStr) async {
    try {
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
      final normalized = await preprocessImage(bytes, _inputSize, _inputSize);

      // Measured off the same buffer the model is about to be fed, so the
      // husk check costs one extra pass over an already-decoded image and
      // judges exactly the pixels the model saw.
      final huskFeatures = HuskGate.analyze(normalized, _inputSize);

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

      return _buildResult(probs, huskFeatures);
    } catch (e, st) {
      AppLogger.e('Inference error', e, st);
      return null;
    }
  }

  /// Re-applies the rejection layer to an already-computed prediction using
  /// the CURRENT thresholds, without re-running the model. This is what
  /// lets the strictness slider live-update an on-screen result: [previous]
  /// already carries the full probability vector, so there's nothing to
  /// recompute except the accept/reject decision itself.
  PredictionResult reevaluate(PredictionResult previous) {
    final probs = [
      for (final label in _labels) previous.classProbabilities[label] ?? 0.0,
    ];
    if (probs.isEmpty || probs.every((p) => p == 0.0)) {
      // Labels changed (e.g. a different model loaded) since previous was
      // computed; nothing sane to re-derive from, leave it as-is.
      return previous;
    }
    // The husk measurements belong to the image, not to the thresholds, so
    // they carry straight over - which is what lets the strictness slider
    // re-decide a result already on screen without re-reading the photo.
    return _buildResult(probs, previous.huskFeatures);
  }

  /// Applies the husk/confidence/margin/centroid-distance rejection layer to
  /// a (normalized) probability vector and builds the resulting
  /// [PredictionResult]. Shared by [predictFromBytes] (fresh inference) and
  /// [reevaluate] (re-deriving a verdict from an already-computed vector at
  /// new thresholds), so the two can never drift out of sync.
  ///
  /// [features] is the image-level husk measurement, or null when there is
  /// none to apply (a history result from before this check existed being
  /// re-evaluated). A null simply skips the husk check rather than failing
  /// it, so re-judging an old result cannot invent a rejection on evidence
  /// that was never recorded.
  PredictionResult _buildResult(List<double> probs, HuskFeatures? features) {
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
    var rejectedBy = RejectionCheck.none;

    // Checked first, and deliberately so. The three probability-based checks
    // below can only ever say "the model was unsure"; none of them can say
    // "that is not a coconut husk", because a non-husk photo makes this
    // model answer with ~100% confidence for one class, which sails through
    // all three (see HuskGate's class doc). Running the image check first
    // also means the user gets the reason that actually applies - "this is
    // not a husk" - instead of a confidence complaint about a photo whose
    // real problem is that it is a picture of a wall.
    if (features != null && features.score < _huskThreshold) {
      rejected = true;
      rejectedBy = RejectionCheck.notHusk;
      rejectionReason = 'Rejected: the image did not pass the husk check - '
          '${features.weakestCue.reason} '
          'It scored ${(features.score * 100).toStringAsFixed(0)}%, below '
          'the ${(_huskThreshold * 100).toStringAsFixed(0)}% minimum, so no '
          'maturity stage was reported.';
    } else if (confidence < _confThreshold) {
      rejected = true;
      rejectedBy = RejectionCheck.confidence;
      rejectionReason = 'Confidence '
          '(${(confidence * 100).toStringAsFixed(1)}%) is below the '
          '${(_confThreshold * 100).toStringAsFixed(0)}% minimum needed '
          'to accept a prediction.';
    } else if (probs.length >= 2 && margin < _marginThreshold) {
      rejected = true;
      rejectedBy = RejectionCheck.margin;
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
        rejectedBy = RejectionCheck.pattern;
        // No longer claims "it is probably not a husk" - the husk check
        // runs before this one and would already have said so. By the time
        // a result reaches here it has passed that check, so what is left
        // is a husk the model cannot place.
        rejectionReason = 'The prediction pattern for this image does not '
            'closely resemble any maturity stage from prior examples. The '
            'husk may be between stages, or the photo unclear.';
      }
    }

    final finalClass = rejected ? 'rejected' : predictedClass;
    return PredictionResult(
      predictedClass: finalClass,
      confidence: confidence,
      explanation: _explanationForClass(finalClass),
      useCases: _useCasesForClass(finalClass),
      classProbabilities: classProbabilities,
      rawPredictedClass: predictedClass,
      margin: margin,
      rejectionReason: rejectionReason,
      huskFeatures: features,
      confThresholdUsed: _confThreshold,
      marginThresholdUsed: _marginThreshold,
      huskThresholdUsed: _huskThreshold,
      rejectedBy: rejectedBy,
    );
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

  /// Decodes and resizes to the model's input, returning raw 0-255 RGB.
  /// Exposed so tests can feed real photos through the same decode/resize
  /// the app uses, rather than through a reimplementation of it that could
  /// agree with the calibration while the real path quietly disagrees.
  @visibleForTesting
  static Future<List<double>> preprocessImage(
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

  String? _useCasesForClass(String predictedClass) {
    switch (predictedClass) {
      case 'immature':
        return 'Best for: coco peat, coir fiber, mulch, compost, soil '
            'conditioner.';
      case 'mature':
        return 'Best for: coir fiber, coco peat, rope/twine, doormats, '
            'floor mats.';
      case 'overmature':
        return 'Best for: coir fiber, coco peat, rope/twine, doormats, '
            'floor mats.';
      default:
        return null;
    }
  }

  void dispose() {
    try {
      _interpreter?.close();
    } catch (_) {}
    _interpreter = null;
  }
}
