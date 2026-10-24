import 'dart:convert';

/// Image-level "does this even look like a coconut husk?" check, applied
/// before the model's own probabilities are trusted.
///
/// ## Why this exists
///
/// The classifier has exactly three outputs - immature, mature, overmature -
/// and softmax forces them to sum to 1. There is no "none of the above", so
/// a photo of a wall, a hand, or the sky does not produce three low scores:
/// it produces one *high* score for whichever class the features happen to
/// lean towards. In practice this model answers non-husk images with ~100%
/// confidence for a single class (see the note in
/// InferenceEngine.preprocessImage, where solid-colour and noise images all
/// came back as 100% "overmature").
///
/// That defeats every rejection rule that only looks at the probability
/// vector: a 100%/0%/0% output passes the confidence floor, passes the
/// margin floor, and sits *exactly on* a class centroid so it passes the
/// centroid-distance check too. The only way to catch a non-husk image is to
/// look at the image itself, which is what this does.
///
/// ## What it measures
///
/// Four cheap statistics over the same 224x224 RGB buffer the model is fed,
/// so there is no extra decode or resize:
///
/// * [warmFraction]   - how much of the frame is husk-coloured (warm hues).
/// * [textureEnergy]  - mean gradient; husk fibre is textured, walls/sky/
///                      screens and out-of-focus shots are not.
/// * [offHueFraction] - how much is blue/purple/magenta, which a coconut
///                      husk never is (sky, clothing, screens, painted walls).
/// * [subjectContrast]- how much more husk-coloured the middle of the frame
///                      is than its border. This is what separates "a husk
///                      placed on a background" from "a photo of a brown
///                      textured surface" such as a wooden table or bark,
///                      which fill the whole frame uniformly.
///
/// Each becomes a 0-1 sub-score through a calibrated ramp, and [score] is
/// the weakest of the four, so one clearly wrong cue is enough to reject.
///
/// ## Calibration
///
/// The ramp bounds below were measured over all 1332 husk photos in the
/// project's `images/` set (299 immature, 333 mature, 329 overmature, plus
/// the 371 held-out "-untrained" photos), against 22 non-husk images (flat
/// and textured surfaces, skin, sky, grass, wood grain, bark, concrete,
/// denim, printed text, noise, plus husk photos deliberately blurred,
/// desaturated and hue-rotated).
///
/// Real husks scored min 0.829, median 1.000. Every non-husk image scored
/// at or below 0.239. The strictness range this feeds (0.25 lenient - 0.65
/// strict, see InferenceEngine) therefore rejected 0 of 1332 real husks
/// while catching all 22 non-husk images.
///
/// ## Known limits
///
/// These are colour/texture/framing statistics, not a trained out-of-
/// distribution detector. They reliably catch what a user actually points a
/// phone at by mistake - people, hands, rooms, floors, walls, sky, plants,
/// screens, print - because none of those put a warm fibrous subject against
/// a plainer surround. They cannot tell a coconut husk from something that
/// genuinely looks like one: a coir doormat, a brown fibrous fruit, or a
/// husk-sized piece of bark photographed the same way would pass this gate
/// and be classified. Closing that gap needs a model trained with a fourth
/// "not a husk" class rather than a front-end check.
///
/// Because the measurement is of the *photo*, not the husk, this also
/// enforces the capture protocol the user manual already asks for: one husk,
/// plain background, in focus. A photo that fills the entire frame with husk
/// and leaves no background visible has little [subjectContrast] and may be
/// rejected - which is the intended trade, since the model was trained
/// exclusively on single husks shot against a plain background and its
/// output is not trustworthy outside that framing.
class HuskFeatures {
  /// Fraction of the frame that is lit and warm-hued (hue 8-75 degrees).
  final double warmFraction;

  /// Mean absolute luma gradient, normalised to 0-1.
  final double textureEnergy;

  /// Fraction of the frame that is a hue a husk never is (170-330 degrees).
  final double offHueFraction;

  /// Warm fraction of the centre minus the warm fraction of the border ring.
  final double subjectContrast;

  const HuskFeatures({
    required this.warmFraction,
    required this.textureEnergy,
    required this.offHueFraction,
    required this.subjectContrast,
  });

  // Ramp bounds: at or below `lo` the sub-score is 0, at or above `hi` it is
  // 1. See the class doc for how these were measured.
  static const double _warmLo = 0.05;
  static const double _warmHi = 0.22;
  static const double _textureLo = 0.012;
  static const double _textureHi = 0.026;
  static const double _offHueLo = 0.06;
  static const double _offHueHi = 0.28;
  static const double _subjectLo = 0.03;
  static const double _subjectHi = 0.12;

  static double _ramp(double x, double lo, double hi) =>
      ((x - lo) / (hi - lo)).clamp(0.0, 1.0);

  /// Inverted ramp, for features where *more* is worse.
  static double _invRamp(double x, double lo, double hi) =>
      ((hi - x) / (hi - lo)).clamp(0.0, 1.0);

  double get warmScore => _ramp(warmFraction, _warmLo, _warmHi);
  double get textureScore => _ramp(textureEnergy, _textureLo, _textureHi);
  double get offHueScore => _invRamp(offHueFraction, _offHueLo, _offHueHi);
  double get subjectScore => _ramp(subjectContrast, _subjectLo, _subjectHi);

  /// Overall 0-1 husk likeness: the weakest of the four sub-scores, so any
  /// single clearly-wrong cue is enough to fail the check.
  double get score {
    double worst = warmScore;
    if (textureScore < worst) worst = textureScore;
    if (offHueScore < worst) worst = offHueScore;
    if (subjectScore < worst) worst = subjectScore;
    return worst;
  }

  /// The sub-score that dragged [score] down, used to explain a rejection in
  /// terms the user can act on rather than just "it failed".
  HuskCue get weakestCue {
    var worst = HuskCue.colour;
    var value = warmScore;
    if (textureScore < value) {
      worst = HuskCue.texture;
      value = textureScore;
    }
    if (offHueScore < value) {
      worst = HuskCue.offHue;
      value = offHueScore;
    }
    if (subjectScore < value) {
      worst = HuskCue.subject;
      value = subjectScore;
    }
    return worst;
  }

  Map<String, dynamic> toJson() => {
        'warm': warmFraction,
        'texture': textureEnergy,
        'off_hue': offHueFraction,
        'subject': subjectContrast,
      };

  static HuskFeatures? fromJson(String? raw) {
    if (raw == null || raw.isEmpty) return null;
    try {
      final m = jsonDecode(raw) as Map<String, dynamic>;
      double read(String k) => (m[k] is num) ? (m[k] as num).toDouble() : 0.0;
      return HuskFeatures(
        warmFraction: read('warm'),
        textureEnergy: read('texture'),
        offHueFraction: read('off_hue'),
        subjectContrast: read('subject'),
      );
    } catch (_) {
      return null;
    }
  }
}

/// Which husk cue was weakest, so the UI can say what to fix.
enum HuskCue { colour, texture, offHue, subject }

extension HuskCueText on HuskCue {
  /// Short label for the decision-reference rows.
  String get label {
    switch (this) {
      case HuskCue.colour:
        return 'Husk colour';
      case HuskCue.texture:
        return 'Fibre detail';
      case HuskCue.offHue:
        return 'Off-colour content';
      case HuskCue.subject:
        return 'Husk against background';
    }
  }

  /// Plain-language reason, phrased as something the user can act on.
  String get reason {
    switch (this) {
      case HuskCue.colour:
        return 'the colours in the photo are not husk colours.';
      case HuskCue.texture:
        return 'the photo is too flat or too blurry to show husk fibre.';
      case HuskCue.offHue:
        return 'too much of the photo is a colour a coconut husk never is.';
      case HuskCue.subject:
        return 'no husk-coloured object stands out from the background - '
            'the frame looks like one surface rather than a husk placed on '
            'a background.';
    }
  }
}

/// Computes [HuskFeatures] from the preprocessed model input.
class HuskGate {
  /// Width of the border ring, in pixels, used as "the background" when
  /// measuring [HuskFeatures.subjectContrast].
  static const int borderRing = 28;

  /// [rgb] is the raw 0-255, row-major, 3-channel interleaved buffer that
  /// InferenceEngine.preprocessImage already built for the model, so this
  /// costs one extra pass over an image that is already decoded and resized.
  static HuskFeatures analyze(List<double> rgb, int size) {
    final expected = size * size * 3;
    if (rgb.length < expected) {
      // Nothing sane to measure; score 0 on every cue so the caller rejects
      // rather than silently accepting an image we could not inspect.
      return const HuskFeatures(
        warmFraction: 0.0,
        textureEnergy: 0.0,
        offHueFraction: 0.0,
        subjectContrast: 0.0,
      );
    }

    final gray = List<double>.filled(size * size, 0.0);
    final warm = List<bool>.filled(size * size, false);

    int warmCount = 0;
    int offHueCount = 0;
    int warmInBorder = 0;
    int warmInCentre = 0;
    int borderPixels = 0;
    int centrePixels = 0;

    for (int y = 0; y < size; y++) {
      final inBorderRow = y < borderRing || y >= size - borderRing;
      for (int x = 0; x < size; x++) {
        final i = y * size + x;
        final o = i * 3;
        final r = rgb[o] / 255.0;
        final g = rgb[o + 1] / 255.0;
        final b = rgb[o + 2] / 255.0;

        gray[i] = 0.299 * rgb[o] + 0.587 * rgb[o + 1] + 0.114 * rgb[o + 2];

        double mx = r;
        if (g > mx) mx = g;
        if (b > mx) mx = b;
        double mn = r;
        if (g < mn) mn = g;
        if (b < mn) mn = b;
        final d = mx - mn;

        final v = mx;
        final s = mx > 0.0 ? d / mx : 0.0;

        double h = 0.0;
        if (d > 1e-12) {
          if (mx == r) {
            h = 60.0 * ((((g - b) / d) % 6.0 + 6.0) % 6.0);
          } else if (mx == g) {
            h = 60.0 * (((b - r) / d) + 2.0);
          } else {
            h = 60.0 * (((r - g) / d) + 4.0);
          }
        }

        final lit = v >= 0.08 && v <= 0.99;
        final isWarm = lit && h >= 8.0 && h <= 75.0 && s >= 0.10;
        final isOffHue = lit && h >= 170.0 && h <= 330.0 && s >= 0.15;

        warm[i] = isWarm;
        if (isWarm) warmCount++;
        if (isOffHue) offHueCount++;

        final inBorder = inBorderRow || x < borderRing || x >= size - borderRing;
        if (inBorder) {
          borderPixels++;
          if (isWarm) warmInBorder++;
        } else {
          centrePixels++;
          if (isWarm) warmInCentre++;
        }
      }
    }

    // Mean of (|dx| + |dy|) / 2 over the (size-1)^2 interior, matching the
    // calibration script's forward-difference definition exactly.
    double gradSum = 0.0;
    final span = size - 1;
    for (int y = 0; y < span; y++) {
      for (int x = 0; x < span; x++) {
        final i = y * size + x;
        final dx = (gray[i + 1] - gray[i]).abs();
        final dy = (gray[i + size] - gray[i]).abs();
        gradSum += (dx + dy) / 2.0;
      }
    }

    final totalPixels = size * size;
    final centreWarm = centrePixels > 0 ? warmInCentre / centrePixels : 0.0;
    final borderWarm = borderPixels > 0 ? warmInBorder / borderPixels : 0.0;

    return HuskFeatures(
      warmFraction: warmCount / totalPixels,
      textureEnergy: gradSum / (span * span) / 255.0,
      offHueFraction: offHueCount / totalPixels,
      subjectContrast: centreWarm - borderWarm,
    );
  }
}
