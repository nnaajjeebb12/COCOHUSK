import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:husktech/husk_gate.dart';
import 'package:husktech/inference_engine.dart';

/// The husk gate's ramp bounds were fitted in Python, which decoded and
/// resized the calibration photos with PIL. The app decodes and resizes with
/// dart:ui instead. The unit tests pin the feature *arithmetic* against that
/// reference, but only real photos through the real decode path can show
/// that the two agree on actual images.
///
/// These photos are the project's training set, which is deliberately not
/// committed (see .gitignore), so this whole file skips when the folder is
/// absent. It is a local calibration guard, not a CI gate.
const _imagesDir = 'images';
const _classDirs = [
  'immature',
  'mature',
  'overmature',
  'immature-untrained',
  'mature-untrained',
  'over-untrained',
];

/// Lowest husk score measured over all 1332 photos during calibration.
const _calibratedMinimum = 0.829;

/// Balanced strictness, where the app starts.
final _balancedThreshold = InferenceEngine.thresholdsAt(0.5).husk;

List<File> _samplePhotos({required int perClass}) {
  final out = <File>[];
  for (final cls in _classDirs) {
    final dir = Directory('$_imagesDir/$cls');
    if (!dir.existsSync()) continue;
    final files = dir
        .listSync()
        .whereType<File>()
        .where((f) =>
            f.path.toLowerCase().endsWith('.jpg') ||
            f.path.toLowerCase().endsWith('.jpeg') ||
            f.path.toLowerCase().endsWith('.png'))
        .toList()
      ..sort((a, b) => a.path.compareTo(b.path));
    if (files.isEmpty) continue;
    // Spread the sample across the folder rather than taking the first few,
    // which in this dataset would all be near-identical shots.
    final step = (files.length / perClass).floor().clamp(1, files.length);
    for (int i = 0, taken = 0; i < files.length && taken < perClass;
        i += step, taken++) {
      out.add(files[i]);
    }
  }
  return out;
}

Future<double> _score(File f) async {
  final rgb = await InferenceEngine.preprocessImage(
      await f.readAsBytes(), 224, 224);
  return HuskGate.analyze(rgb, 224).score;
}

void main() {
  final hasPhotos = Directory(_imagesDir).existsSync() &&
      _classDirs.any((c) => Directory('$_imagesDir/$c').existsSync());

  group('real husk photos through the app decode path', () {
    testWidgets('every sampled husk photo passes the balanced husk check',
        (tester) async {
      final photos = _samplePhotos(perClass: 6);
      expect(photos, isNotEmpty);

      final scores = <String, double>{};
      // Decoding goes through the real engine, so it needs the real event
      // loop rather than the fake async the widget tester installs.
      await tester.runAsync(() async {
        for (final f in photos) {
          scores[f.path.split(RegExp(r'[\\/]')).last] = await _score(f);
        }
      });

      final failures = scores.entries
          .where((e) => e.value < _balancedThreshold)
          .map((e) => '${e.key}=${e.value.toStringAsFixed(3)}')
          .toList();

      expect(failures, isEmpty,
          reason: 'these real husk photos would now be rejected as '
              'non-husk: ${failures.join(", ")}');
    }, skip: !hasPhotos);

    testWidgets('scores stay in the range the thresholds were fitted to',
        (tester) async {
      // Guards against the dart:ui resize drifting from the PIL resize the
      // bounds were fitted with. A small gap is expected; a large one means
      // the calibration no longer describes what the app measures.
      final photos = _samplePhotos(perClass: 6);
      expect(photos, isNotEmpty);

      var lowest = 1.0;
      await tester.runAsync(() async {
        for (final f in photos) {
          final s = await _score(f);
          if (s < lowest) lowest = s;
        }
      });

      expect(lowest, greaterThan(_calibratedMinimum - 0.15),
          reason: 'lowest real-photo husk score was '
              '${lowest.toStringAsFixed(3)}, well under the '
              '$_calibratedMinimum measured during calibration');
    }, skip: !hasPhotos);
  });
}
