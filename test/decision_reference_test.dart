import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:husktech/decision_reference.dart';
import 'package:husktech/husk_gate.dart';
import 'package:husktech/inference_engine.dart';
import 'package:husktech/main.dart';

/// Balanced strictness, the values the engine actually starts at.
final _balanced = InferenceEngine.thresholdsAt(0.5);

PredictionResult _rejectedAsNotHusk() => PredictionResult(
      predictedClass: 'rejected',
      confidence: 0.9998,
      explanation: 'Rejected: the image does not sufficiently match any '
          'known coconut husk maturity stage.',
      classProbabilities: const {
        'immature': 0.0001,
        'mature': 0.0001,
        'overmature': 0.9998,
      },
      rawPredictedClass: 'overmature',
      margin: 0.9997,
      rejectionReason: 'Rejected: the image did not pass the husk check - '
          'no husk-coloured object stands out from the background.',
      huskFeatures: const HuskFeatures(
        warmFraction: 0.98,
        textureEnergy: 0.045,
        offHueFraction: 0.0,
        subjectContrast: 0.0,
      ),
      confThresholdUsed: _balanced.confidence,
      marginThresholdUsed: _balanced.margin,
      huskThresholdUsed: _balanced.husk,
      rejectedBy: RejectionCheck.notHusk,
    );

PredictionResult _accepted() => PredictionResult(
      predictedClass: 'mature',
      confidence: 0.9999,
      explanation: 'Mature: the ideal harvest stage.',
      useCases: 'Best for: coir fiber, coco peat.',
      classProbabilities: const {
        'immature': 0.00005,
        'mature': 0.9999,
        'overmature': 0.00005,
      },
      margin: 0.99985,
      huskFeatures: const HuskFeatures(
        warmFraction: 0.70,
        textureEnergy: 0.047,
        offHueFraction: 0.0,
        subjectContrast: 0.55,
      ),
      confThresholdUsed: _balanced.confidence,
      marginThresholdUsed: _balanced.margin,
      huskThresholdUsed: _balanced.husk,
    );

Future<void> _pump(WidgetTester tester, Widget child, double width) async {
  tester.view.physicalSize = Size(width, 900);
  tester.view.devicePixelRatio = 1.0;
  addTearDown(tester.view.reset);
  await tester.pumpWidget(MaterialApp(
    home: Scaffold(
      body: SingleChildScrollView(child: Padding(
        padding: const EdgeInsets.all(16),
        child: child,
      )),
    ),
  ));
  await tester.pumpAndSettle();
}

void main() {
  group('DecisionReference', () {
    testWidgets('shows every check with its measured value and minimum',
        (tester) async {
      await _pump(tester, DecisionReference(prediction: _accepted()), 360);

      expect(find.text('Decision reference'), findsOneWidget);
      expect(find.text('Husk check'), findsOneWidget);
      expect(find.text('Confidence'), findsOneWidget);
      expect(find.text('Margin'), findsOneWidget);
      // The threshold is printed next to the measurement, which is the
      // whole point of the client's "reference ng threshold" request.
      expect(find.text('min ${(_balanced.husk * 100).round()}%'),
          findsOneWidget);
      expect(find.text('min ${(_balanced.confidence * 100).round()}%'),
          findsOneWidget);
    });

    testWidgets('an accepted scan passes all three checks', (tester) async {
      await _pump(tester, DecisionReference(prediction: _accepted()), 360);
      expect(find.byIcon(Icons.check_circle), findsNWidgets(3));
      expect(find.byIcon(Icons.cancel), findsNothing);
    });

    testWidgets('a non-husk scan fails only the husk check', (tester) async {
      await _pump(
          tester, DecisionReference(prediction: _rejectedAsNotHusk()), 360);

      // Confidence and margin both pass - which is exactly why they could
      // never have caught this image on their own.
      expect(find.byIcon(Icons.check_circle), findsNWidgets(2));
      expect(find.byIcon(Icons.cancel), findsOneWidget);
      expect(find.textContaining('Husk against background'), findsWidgets);
    });

    testWidgets('renders nothing when no thresholds were recorded',
        (tester) async {
      // Scans saved before schema v2 have no reference to show; showing
      // zeroes would be worse than showing nothing.
      final legacy = PredictionResult(
        predictedClass: 'mature',
        confidence: 0.99,
        explanation: 'x',
      );
      await _pump(tester, DecisionReference(prediction: legacy), 360);
      expect(find.text('Decision reference'), findsNothing);
    });

    testWidgets('omits the husk row for results that never had one',
        (tester) async {
      final noFeatures = PredictionResult(
        predictedClass: 'mature',
        confidence: 0.99,
        explanation: 'x',
        margin: 0.98,
        confThresholdUsed: _balanced.confidence,
        marginThresholdUsed: _balanced.margin,
        huskThresholdUsed: _balanced.husk,
      );
      await _pump(tester, DecisionReference(prediction: noFeatures), 360);
      expect(find.text('Decision reference'), findsOneWidget);
      expect(find.text('Husk check'), findsNothing);
      expect(find.text('Confidence'), findsOneWidget);
    });

    for (final width in const [320.0, 360.0, 412.0]) {
      testWidgets('lays out without overflow at ${width.toInt()}dp',
          (tester) async {
        await _pump(
            tester, DecisionReference(prediction: _rejectedAsNotHusk()), width);
        expect(tester.takeException(), isNull);
      });
    }
  });

  group('Information page threshold reference', () {
    for (final width in const [320.0, 360.0, 412.0]) {
      testWidgets('renders the preset table at ${width.toInt()}dp',
          (tester) async {
        tester.view.physicalSize = Size(width, 900);
        tester.view.devicePixelRatio = 1.0;
        addTearDown(tester.view.reset);

        await tester.pumpWidget(const MaterialApp(home: InfoPage()));
        await tester.pumpAndSettle();

        // The page is a lazy ListView, so scrolling is also what builds and
        // lays out every card on the way down - which is what makes this an
        // overflow check for the whole page, not just its first screen.
        await tester.scrollUntilVisible(
          find.text('Threshold reference'),
          300,
          scrollable: find.byType(Scrollable).first,
        );

        expect(find.text('Threshold reference'), findsOneWidget);
        expect(find.text('Balanced (default)'), findsOneWidget);
        expect(find.text('Very Strict'), findsOneWidget);
        expect(tester.takeException(), isNull);
      });
    }

    testWidgets('the printed table matches what the engine applies',
        (tester) async {
      tester.view.physicalSize = const Size(400, 1400);
      tester.view.devicePixelRatio = 1.0;
      addTearDown(tester.view.reset);

      await tester.pumpWidget(const MaterialApp(home: InfoPage()));
      await tester.pumpAndSettle();

      // A hard-coded table would quietly go stale the next time the bounds
      // are recalibrated; this asserts the page is reading the real values.
      for (final strictness in const [0.0, 0.5, 1.0]) {
        final t = InferenceEngine.thresholdsAt(strictness);
        await tester.scrollUntilVisible(
          find.text('${(t.husk * 100).round()}%'),
          200,
          scrollable: find.byType(Scrollable).first,
        );
        expect(find.text('${(t.husk * 100).round()}%'), findsWidgets);
      }
    });
  });
}
