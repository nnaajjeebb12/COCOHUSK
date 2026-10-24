import 'package:flutter_test/flutter_test.dart';
import 'package:husktech/husk_gate.dart';
import 'package:husktech/inference_engine.dart';

const int _size = 224;

/// Builds the 0-255, row-major, 3-channel interleaved buffer the gate reads,
/// the same layout InferenceEngine._preprocessImage produces.
List<double> _buffer(List<int> Function(int x, int y) pixel) {
  final out = List<double>.filled(_size * _size * 3, 0.0);
  for (int y = 0; y < _size; y++) {
    for (int x = 0; x < _size; x++) {
      final rgb = pixel(x, y);
      final o = (y * _size + x) * 3;
      out[o] = rgb[0].toDouble();
      out[o + 1] = rgb[1].toDouble();
      out[o + 2] = rgb[2].toDouble();
    }
  }
  return out;
}

List<double> get _flatTan => _buffer((x, y) => const [165, 125, 85]);
List<double> get _flatGrey => _buffer((x, y) => const [150, 150, 150]);
List<double> get _blueScene =>
    _buffer((x, y) => [(x * 3) % 40, (y * 3) % 40, 200]);

/// A warm, finely textured subject centred on a plain near-white surround -
/// the shape of a correctly captured husk photo.
List<double> get _blobOnWhite => _buffer((x, y) {
      final inBlob = x >= 56 && x < 168 && y >= 56 && y < 168;
      if (!inBlob) return const [250, 250, 248];
      return [
        130 + ((x * 37 + y * 53) % 90),
        95 + ((x * 29 + y * 41) % 70),
        60 + ((x * 23 + y * 59) % 50),
      ];
    });

void main() {
  // The ramp bounds in HuskFeatures were calibrated in Python over the
  // project's 1332 husk photos. These expectations are that script's output
  // for four patterns both languages generate with identical integer
  // arithmetic, so they fail if the Dart port of the feature maths ever
  // drifts from what the thresholds were actually fitted to.
  group('feature maths matches the calibration reference', () {
    void expectFeatures(
      String name,
      HuskFeatures f, {
      required double warm,
      required double texture,
      required double offHue,
      required double subject,
    }) {
      expect(f.warmFraction, closeTo(warm, 1e-9), reason: '$name warm');
      expect(f.textureEnergy, closeTo(texture, 1e-9), reason: '$name texture');
      expect(f.offHueFraction, closeTo(offHue, 1e-9), reason: '$name offHue');
      expect(f.subjectContrast, closeTo(subject, 1e-9), reason: '$name subject');
    }

    test('flat tan', () {
      expectFeatures('flatTan', HuskGate.analyze(_flatTan, _size),
          warm: 1.0, texture: 0.0, offHue: 0.0, subject: 0.0);
    });

    test('flat grey', () {
      expectFeatures('flatGrey', HuskGate.analyze(_flatGrey, _size),
          warm: 0.0, texture: 0.0, offHue: 0.0, subject: 0.0);
    });

    test('blue scene', () {
      expectFeatures('blueScene', HuskGate.analyze(_blueScene, _size),
          warm: 0.0, texture: 0.009449732, offHue: 1.0, subject: 0.0);
    });

    test('textured warm subject on a plain surround', () {
      expectFeatures('blobOnWhite', HuskGate.analyze(_blobOnWhite, _size),
          warm: 0.220384247,
          texture: 0.028086333,
          offHue: 0.0,
          subject: 0.391794218);
    });
  });

  group('gate decisions', () {
    // 0.25 lenient .. 0.65 strict, the range InferenceEngine interpolates.
    const lenient = 0.25;
    const strict = 0.65;

    test('a husk-shaped capture passes at every strictness', () {
      final f = HuskGate.analyze(_blobOnWhite, _size);
      expect(f.score, greaterThan(strict));
    });

    test('husk colour alone is not enough without fibre detail', () {
      // The exact case the probability-only checks could never catch: the
      // right colour, but a flat surface rather than a husk.
      final f = HuskGate.analyze(_flatTan, _size);
      expect(f.warmScore, 1.0);
      expect(f.score, lessThan(lenient));
      expect(f.weakestCue, isIn([HuskCue.texture, HuskCue.subject]));
    });

    test('a flat grey surface fails on colour', () {
      final f = HuskGate.analyze(_flatGrey, _size);
      expect(f.score, lessThan(lenient));
      expect(f.weakestCue, HuskCue.colour);
    });

    test('a blue scene fails', () {
      final f = HuskGate.analyze(_blueScene, _size);
      expect(f.score, lessThan(lenient));
      expect(f.offHueFraction, 1.0);
    });

    test('a buffer too small to measure scores zero rather than passing', () {
      final f = HuskGate.analyze(const <double>[1.0, 2.0, 3.0], _size);
      expect(f.score, 0.0);
    });
  });

  group('threshold reference', () {
    test('every threshold tightens monotonically with strictness', () {
      final lenient = InferenceEngine.thresholdsAt(0.0);
      final balanced = InferenceEngine.thresholdsAt(0.5);
      final strict = InferenceEngine.thresholdsAt(1.0);

      expect(lenient.confidence, lessThan(balanced.confidence));
      expect(balanced.confidence, lessThan(strict.confidence));
      expect(lenient.margin, lessThan(balanced.margin));
      expect(balanced.margin, lessThan(strict.margin));
      expect(lenient.husk, lessThan(balanced.husk));
      expect(balanced.husk, lessThan(strict.husk));
    });

    test('the husk range sits in the gap measured between husk and non-husk',
        () {
      // Real husks scored >= 0.829 and non-husk images <= 0.239 during
      // calibration; both ends of the slider must stay inside that gap or
      // the slider starts trading real rejections for real misses.
      final lenient = InferenceEngine.thresholdsAt(0.0);
      final strict = InferenceEngine.thresholdsAt(1.0);
      expect(lenient.husk, greaterThan(0.239));
      expect(strict.husk, lessThan(0.829));
    });

    test('strictness is clamped to 0..1', () {
      expect(InferenceEngine.thresholdsAt(-5).husk,
          InferenceEngine.thresholdsAt(0.0).husk);
      expect(InferenceEngine.thresholdsAt(5).husk,
          InferenceEngine.thresholdsAt(1.0).husk);
    });
  });

  group('husk features survive a round trip through history', () {
    test('encode then decode preserves every measurement', () {
      final original = HuskGate.analyze(_blobOnWhite, _size);
      final restored = HuskFeatures.fromJson(
          '{"warm":${original.warmFraction},'
          '"texture":${original.textureEnergy},'
          '"off_hue":${original.offHueFraction},'
          '"subject":${original.subjectContrast}}');
      expect(restored, isNotNull);
      expect(restored!.score, closeTo(original.score, 1e-12));
    });

    test('a row with no recorded features decodes to null, not zeroes', () {
      // Old history rows must not be re-judged as "not a husk" on evidence
      // that was never recorded for them.
      expect(HuskFeatures.fromJson(null), isNull);
      expect(HuskFeatures.fromJson(''), isNull);
      expect(HuskFeatures.fromJson('not json'), isNull);
    });
  });

  group('threshold reference on a result', () {
    test('a result without recorded thresholds reports no reference', () {
      final p = PredictionResult(
        predictedClass: 'mature',
        confidence: 0.99,
        explanation: 'x',
      );
      expect(p.hasThresholdReference, isFalse);
      expect(p.rejectedBy, RejectionCheck.none);
    });

    test('a result with recorded thresholds reports a reference', () {
      final p = PredictionResult(
        predictedClass: 'rejected',
        confidence: 0.99,
        explanation: 'x',
        confThresholdUsed: 0.9245,
        marginThresholdUsed: 0.895,
        huskThresholdUsed: 0.45,
        rejectedBy: RejectionCheck.notHusk,
      );
      expect(p.hasThresholdReference, isTrue);
    });
  });
}
