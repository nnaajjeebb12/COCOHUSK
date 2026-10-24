import 'package:flutter/material.dart';

import 'husk_gate.dart';
import 'inference_engine.dart';

/// The "why this verdict" block: every check a scan had to pass, what this
/// image actually measured, and the threshold it was measured against.
///
/// The thresholds come from the [PredictionResult] itself rather than from
/// the live engine, so a scan opened from history shows the reference values
/// it was judged by at the time, even if the strictness slider has moved
/// since. Results saved before those values were recorded report
/// [PredictionResult.hasThresholdReference] as false and this renders
/// nothing, rather than showing a row of zeroes.
class DecisionReference extends StatelessWidget {
  final PredictionResult prediction;

  /// When false, the per-cue breakdown under a failed husk check is left
  /// out. Used on the home card, where space is tight and the rejection
  /// reason above it already names the cue in words.
  final bool showHuskCues;

  const DecisionReference({
    super.key,
    required this.prediction,
    this.showHuskCues = true,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    if (!prediction.hasThresholdReference) return const SizedBox.shrink();

    final muted = theme.colorScheme.onSurface.withValues(alpha: 0.6);
    final features = prediction.huskFeatures;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Icon(Icons.rule, size: 16, color: muted),
            const SizedBox(width: 6),
            Expanded(
              child: Text(
                'Decision reference',
                style: theme.textTheme.titleSmall
                    ?.copyWith(fontWeight: FontWeight.w700),
              ),
            ),
          ],
        ),
        const SizedBox(height: 2),
        Text(
          'What this scan measured, and the threshold it had to reach.',
          style: theme.textTheme.bodySmall?.copyWith(color: muted),
        ),
        const SizedBox(height: 10),
        if (features != null)
          _CheckRow(
            label: 'Husk check',
            measured: features.score,
            threshold: prediction.huskThresholdUsed,
            failed: prediction.rejectedBy == RejectionCheck.notHusk,
            detail: showHuskCues && !_passes(features.score, prediction.huskThresholdUsed)
                ? 'Weakest cue: ${features.weakestCue.label}'
                : null,
          ),
        _CheckRow(
          label: 'Confidence',
          measured: prediction.confidence,
          threshold: prediction.confThresholdUsed,
          failed: prediction.rejectedBy == RejectionCheck.confidence,
        ),
        _CheckRow(
          label: 'Margin',
          measured: prediction.margin,
          threshold: prediction.marginThresholdUsed,
          failed: prediction.rejectedBy == RejectionCheck.margin,
        ),
        if (prediction.rejectedBy == RejectionCheck.pattern) ...[
          const SizedBox(height: 6),
          Text(
            'Pattern check: the probabilities did not resemble any class '
            'seen in training.',
            style: theme.textTheme.bodySmall
                ?.copyWith(color: theme.colorScheme.error),
          ),
        ],
        if (features != null && showHuskCues) ...[
          const SizedBox(height: 10),
          _HuskCueBreakdown(features: features),
        ],
      ],
    );
  }

  static bool _passes(double measured, double threshold) =>
      measured >= threshold;
}

/// One check: measured value, the minimum it had to reach, and whether it
/// got there. The check that actually caused a rejection is marked, so the
/// user can see which line decided the verdict instead of comparing numbers
/// by eye.
class _CheckRow extends StatelessWidget {
  final String label;
  final double measured;
  final double threshold;
  final bool failed;
  final String? detail;

  const _CheckRow({
    required this.label,
    required this.measured,
    required this.threshold,
    required this.failed,
    this.detail,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final passed = measured >= threshold;
    final muted = theme.colorScheme.onSurface.withValues(alpha: 0.6);
    final color = passed ? theme.colorScheme.primary : theme.colorScheme.error;

    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(passed ? Icons.check_circle : Icons.cancel,
                  size: 16, color: color),
              const SizedBox(width: 6),
              Expanded(
                child: Text(
                  label,
                  overflow: TextOverflow.ellipsis,
                  style: theme.textTheme.bodyMedium?.copyWith(
                    fontWeight: failed ? FontWeight.w700 : FontWeight.normal,
                  ),
                ),
              ),
              const SizedBox(width: 8),
              Text(
                '${(measured * 100).toStringAsFixed(1)}%',
                style: theme.textTheme.bodyMedium
                    ?.copyWith(color: color, fontWeight: FontWeight.w600),
              ),
              const SizedBox(width: 6),
              Text(
                'min ${(threshold * 100).toStringAsFixed(0)}%',
                style: theme.textTheme.bodySmall?.copyWith(color: muted),
              ),
            ],
          ),
          const SizedBox(height: 4),
          ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: Stack(
              children: [
                LinearProgressIndicator(
                  value: measured.clamp(0.0, 1.0),
                  minHeight: 6,
                  backgroundColor:
                      theme.colorScheme.onSurface.withValues(alpha: 0.08),
                  valueColor: AlwaysStoppedAnimation<Color>(color),
                ),
                // The threshold itself, drawn on the bar so "how close was
                // it" is readable without doing the subtraction.
                Positioned.fill(
                  child: FractionallySizedBox(
                    alignment: Alignment.centerLeft,
                    widthFactor: threshold.clamp(0.0, 1.0),
                    child: Align(
                      alignment: Alignment.centerRight,
                      child: Container(width: 2, color: muted),
                    ),
                  ),
                ),
              ],
            ),
          ),
          if (detail != null) ...[
            const SizedBox(height: 4),
            Text(
              detail!,
              style: theme.textTheme.bodySmall?.copyWith(color: muted),
            ),
          ],
        ],
      ),
    );
  }
}

/// The four image cues behind the husk check, each as its own 0-100% bar.
/// Shown where there is room for it, so a rejected scan can be diagnosed
/// ("the photo was too blurry" vs "nothing stood out from the background")
/// rather than just reported.
class _HuskCueBreakdown extends StatelessWidget {
  final HuskFeatures features;

  const _HuskCueBreakdown({required this.features});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final muted = theme.colorScheme.onSurface.withValues(alpha: 0.6);
    final rows = <(String, double)>[
      (HuskCue.colour.label, features.warmScore),
      (HuskCue.texture.label, features.textureScore),
      (HuskCue.offHue.label, features.offHueScore),
      (HuskCue.subject.label, features.subjectScore),
    ];

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Husk check, cue by cue',
          style:
              theme.textTheme.bodySmall?.copyWith(fontWeight: FontWeight.w700),
        ),
        const SizedBox(height: 2),
        Text(
          'The husk score above is the weakest of these four.',
          style: theme.textTheme.bodySmall?.copyWith(color: muted),
        ),
        const SizedBox(height: 8),
        for (final (label, value) in rows)
          Padding(
            padding: const EdgeInsets.only(bottom: 6),
            child: Row(
              children: [
                Expanded(
                  child: Text(
                    label,
                    overflow: TextOverflow.ellipsis,
                    style: theme.textTheme.bodySmall,
                  ),
                ),
                const SizedBox(width: 8),
                SizedBox(
                  width: 90,
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(4),
                    child: LinearProgressIndicator(
                      value: value.clamp(0.0, 1.0),
                      minHeight: 6,
                      backgroundColor:
                          theme.colorScheme.onSurface.withValues(alpha: 0.08),
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                SizedBox(
                  width: 40,
                  child: Text(
                    '${(value * 100).toStringAsFixed(0)}%',
                    textAlign: TextAlign.right,
                    style: theme.textTheme.bodySmall?.copyWith(color: muted),
                  ),
                ),
              ],
            ),
          ),
      ],
    );
  }
}
