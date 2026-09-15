import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';

import 'history_store.dart';
import 'main.dart' show classBadgeColor, formatClassName;

/// The classes plotted on the trends chart, in a fixed, stable order so
/// each class always gets the same line/legend position across reloads.
const List<String> _chartClasses = [
  'immature',
  'mature',
  'overmature',
  'rejected'
];

const List<String> _monthShort = [
  'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
];
const List<String> _monthLong = [
  'January', 'February', 'March', 'April', 'May', 'June',
  'July', 'August', 'September', 'October', 'November', 'December',
];
const List<String> _weekdayShort = [
  'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'
];

/// Day/Week/Month/Year filtered trends view: one line per class, a tooltip
/// with the bucket's date/time, and an analysis block underneath. Reads
/// come from [HistoryStore], already scoped to this device's own data.
class HistoryTrendsView extends StatefulWidget {
  /// Where buckets come from. Defaults to [HistoryStore.instance.aggregate];
  /// injectable so the view can be rendered with known data in tests.
  final Future<List<HistoryBucket>> Function(HistoryRange range)? loader;

  const HistoryTrendsView({super.key, this.loader});

  @override
  State<HistoryTrendsView> createState() => _HistoryTrendsViewState();
}

class _HistoryTrendsViewState extends State<HistoryTrendsView> {
  HistoryRange _range = HistoryRange.day;
  late Future<List<HistoryBucket>> _future;

  Future<List<HistoryBucket>> _load(HistoryRange range) =>
      (widget.loader ?? HistoryStore.instance.aggregate)(range);

  @override
  void initState() {
    super.initState();
    _future = _load(_range);
  }

  void _setRange(HistoryRange range) {
    if (range == _range) return;
    setState(() {
      _range = range;
      _future = _load(_range);
    });
  }

  @override
  Widget build(BuildContext context) {
    // Whole tab scrolls: the chart is a fixed-height block and the analysis
    // below it can grow, so nothing gets squeezed on a short phone screen.
    return FutureBuilder<List<HistoryBucket>>(
      future: _future,
      builder: (context, snapshot) {
        final loading = snapshot.connectionState != ConnectionState.done;
        final buckets = snapshot.data ?? const [];
        final totals = _totals(buckets);
        final total = totals.values.fold<int>(0, (a, b) => a + b);
        return ListView(
          padding: const EdgeInsets.all(16),
          children: [
            _buildRangeSelector(context),
            const SizedBox(height: 16),
            SizedBox(
              height: 260,
              child: loading
                  ? const Center(child: CircularProgressIndicator())
                  : _buildChart(context, buckets),
            ),
            const SizedBox(height: 16),
            if (!loading) _buildTotals(context, totals, total),
            const SizedBox(height: 16),
            if (!loading) _buildAnalysis(context, buckets, totals, total),
          ],
        );
      },
    );
  }

  Widget _buildRangeSelector(BuildContext context) {
    return SegmentedButton<HistoryRange>(
      segments: const [
        ButtonSegment(value: HistoryRange.day, label: Text('Day')),
        ButtonSegment(value: HistoryRange.week, label: Text('Week')),
        ButtonSegment(value: HistoryRange.month, label: Text('Month')),
        ButtonSegment(value: HistoryRange.year, label: Text('Year')),
      ],
      selected: {_range},
      onSelectionChanged: (selection) => _setRange(selection.first),
    );
  }

  // ---------------------------------------------------------------- chart

  Widget _buildChart(BuildContext context, List<HistoryBucket> buckets) {
    if (buckets.isEmpty) return const SizedBox.shrink();
    final theme = Theme.of(context);

    final maxCount = buckets
        .expand((b) => b.countsByClass.values)
        .fold<int>(0, (a, b) => a > b ? a : b);
    // Keep a sensible y-axis even with an empty or near-empty range, rather
    // than fl_chart dividing by a maxY of 0.
    final maxY = (maxCount < 4 ? 4 : maxCount + 1).toDouble();

    return LineChart(
      LineChartData(
        minY: 0,
        maxY: maxY,
        // Give the line a little room on both ends so a point on the first
        // or last bucket doesn't sit on top of the border.
        minX: -0.3,
        maxX: buckets.length - 1 + 0.3,
        clipData: const FlClipData.all(),
        gridData: const FlGridData(show: true, drawVerticalLine: false),
        borderData: FlBorderData(
          show: true,
          border: Border.all(color: theme.dividerColor),
        ),
        titlesData: FlTitlesData(
          topTitles:
              const AxisTitles(sideTitles: SideTitles(showTitles: false)),
          rightTitles:
              const AxisTitles(sideTitles: SideTitles(showTitles: false)),
          leftTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              reservedSize: 30,
              interval: maxY > 8 ? (maxY / 4).roundToDouble() : 1,
              getTitlesWidget: (value, meta) => Text(
                value.toInt().toString(),
                style: const TextStyle(fontSize: 10),
              ),
            ),
          ),
          bottomTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              reservedSize: 24,
              interval: 1,
              getTitlesWidget: (value, meta) {
                // Only label real bucket positions, never the padding
                // added by minX/maxX.
                if ((value - value.roundToDouble()).abs() > 0.01) {
                  return const SizedBox.shrink();
                }
                final i = value.round();
                if (i < 0 || i >= buckets.length) {
                  return const SizedBox.shrink();
                }
                final label = _axisLabel(buckets, i);
                if (label.isEmpty) return const SizedBox.shrink();
                return Padding(
                  padding: const EdgeInsets.only(top: 4),
                  child: Text(label, style: const TextStyle(fontSize: 10)),
                );
              },
            ),
          ),
        ),
        lineTouchData: LineTouchData(
          touchTooltipData: LineTouchTooltipData(
            // Keep the tooltip inside the chart's own bounds instead of
            // letting it spill over the top/side border.
            fitInsideHorizontally: true,
            fitInsideVertically: true,
            tooltipMargin: 8,
            tooltipPadding:
                const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
            getTooltipColor: (_) => theme.colorScheme.inverseSurface,
            getTooltipItems: (spots) {
              if (spots.isEmpty) return const [];
              final i = spots.first.x.round().clamp(0, buckets.length - 1);
              final header = _tooltipHeader(buckets[i]);
              final onInverse = theme.colorScheme.onInverseSurface;
              return spots.map((s) {
                final className = _chartClasses[s.barIndex];
                final color = classBadgeColor(className, Colors.grey);
                final line =
                    '${formatClassName(className)}: ${s.y.toInt()}';
                // Date/time header rides on the first item only, so it is
                // shown once at the top of the tooltip.
                if (s == spots.first) {
                  return LineTooltipItem(
                    header,
                    TextStyle(
                      color: onInverse,
                      fontWeight: FontWeight.w600,
                      fontSize: 11,
                    ),
                    textAlign: TextAlign.left,
                    children: [
                      TextSpan(
                        text: '\n$line',
                        style: TextStyle(
                          color: color,
                          fontWeight: FontWeight.w500,
                          fontSize: 12,
                        ),
                      ),
                    ],
                  );
                }
                return LineTooltipItem(
                  line,
                  TextStyle(
                    color: color,
                    fontWeight: FontWeight.w500,
                    fontSize: 12,
                  ),
                  textAlign: TextAlign.left,
                );
              }).toList();
            },
          ),
        ),
        lineBarsData: _chartClasses.map((className) {
          final color = classBadgeColor(className, Colors.grey);
          return LineChartBarData(
            spots: List.generate(
              buckets.length,
              (i) => FlSpot(
                i.toDouble(),
                (buckets[i].countsByClass[className] ?? 0).toDouble(),
              ),
            ),
            isCurved: false,
            color: color,
            barWidth: 2,
            dotData: FlDotData(show: buckets.length <= 12),
            belowBarData: BarAreaData(show: false),
          );
        }).toList(),
      ),
    );
  }

  // --------------------------------------------------------------- totals

  Widget _buildTotals(
      BuildContext context, Map<String, int> totals, int total) {
    return Wrap(
      spacing: 12,
      runSpacing: 8,
      children: [
        for (final className in _chartClasses)
          _chip(
            formatClassName(className),
            totals[className] ?? 0,
            classBadgeColor(className, Colors.grey),
          ),
        _chip('Total', total, Theme.of(context).colorScheme.primary),
      ],
    );
  }

  Widget _chip(String label, int count, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: color.withValues(alpha: 0.4)),
      ),
      child: Text(
        '$label: $count',
        style:
            TextStyle(fontSize: 12, fontWeight: FontWeight.w600, color: color),
      ),
    );
  }

  // ------------------------------------------------------------- analysis

  Widget _buildAnalysis(
    BuildContext context,
    List<HistoryBucket> buckets,
    Map<String, int> totals,
    int total,
  ) {
    final theme = Theme.of(context);
    final muted = theme.colorScheme.onSurface.withValues(alpha: 0.65);

    if (total == 0) {
      return Card(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Text(
            'No scans ${_rangeNoun()} yet. Scan a husk and the analysis '
            'will appear here.',
            style: theme.textTheme.bodyMedium?.copyWith(color: muted),
          ),
        ),
      );
    }

    // Dominant real stage (rejected is reported separately as a rate).
    final stageTotals = {
      for (final c in _chartClasses)
        if (c != 'rejected') c: totals[c] ?? 0,
    };
    final accepted = stageTotals.values.fold<int>(0, (a, b) => a + b);
    final rejected = totals['rejected'] ?? 0;
    String? dominant;
    int dominantCount = 0;
    stageTotals.forEach((c, n) {
      if (n > dominantCount) {
        dominant = c;
        dominantCount = n;
      }
    });

    // Average confidence over accepted scans.
    double confSum = 0;
    for (final b in buckets) {
      b.confidenceSumByClass.forEach((c, v) {
        if (c != 'rejected') confSum += v;
      });
    }
    final avgConf = accepted == 0 ? null : confSum / accepted;

    // Busiest bucket.
    HistoryBucket? busiest;
    for (final b in buckets) {
      if (busiest == null || b.total > busiest.total) busiest = b;
    }
    final activeBuckets = buckets.where((b) => b.total > 0).length;

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Analysis (${_rangeNoun()})',
                style: theme.textTheme.titleMedium),
            const SizedBox(height: 12),
            _fact(
              theme,
              Icons.photo_library_outlined,
              'Total scans',
              '$total across $activeBuckets ${_bucketNoun(activeBuckets)}',
            ),
            if (dominant != null && dominantCount > 0)
              _fact(
                theme,
                Icons.star_outline,
                'Most common stage',
                '${formatClassName(dominant!)}, $dominantCount of $accepted '
                    'accepted (${_pct(dominantCount, accepted)})',
                color: classBadgeColor(dominant!, theme.colorScheme.primary),
              ),
            _fact(
              theme,
              Icons.verified_outlined,
              'Accepted vs rejected',
              '$accepted accepted (${_pct(accepted, total)}), '
                  '$rejected rejected (${_pct(rejected, total)})',
            ),
            if (avgConf != null)
              _fact(
                theme,
                Icons.speed_outlined,
                'Average confidence',
                '${(avgConf * 100).toStringAsFixed(1)}% on accepted scans',
              ),
            if (busiest != null && busiest.total > 0)
              _fact(
                theme,
                Icons.schedule_outlined,
                _busiestLabel(),
                '${_tooltipHeader(busiest)} (${busiest.total} '
                    '${busiest.total == 1 ? "scan" : "scans"})',
              ),
            const SizedBox(height: 12),
            Text('Stage distribution',
                style: theme.textTheme.titleSmall
                    ?.copyWith(fontWeight: FontWeight.w700)),
            const SizedBox(height: 8),
            for (final className in _chartClasses)
              _distributionRow(
                theme,
                className,
                totals[className] ?? 0,
                total,
              ),
          ],
        ),
      ),
    );
  }

  Widget _fact(ThemeData theme, IconData icon, String label, String value,
      {Color? color}) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, size: 18, color: color ?? theme.colorScheme.primary),
          const SizedBox(width: 8),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(label,
                    style: theme.textTheme.bodySmall?.copyWith(
                      color:
                          theme.colorScheme.onSurface.withValues(alpha: 0.65),
                    )),
                Text(value, style: theme.textTheme.bodyMedium),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _distributionRow(
      ThemeData theme, String className, int count, int total) {
    final color = classBadgeColor(className, theme.colorScheme.primary);
    final fraction = total == 0 ? 0.0 : count / total;
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(
                  formatClassName(className),
                  style: theme.textTheme.bodySmall
                      ?.copyWith(fontWeight: FontWeight.w600, color: color),
                  overflow: TextOverflow.ellipsis,
                ),
              ),
              Text(
                '$count (${_pct(count, total)})',
                style: theme.textTheme.bodySmall,
              ),
            ],
          ),
          const SizedBox(height: 4),
          ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: LinearProgressIndicator(
              value: fraction,
              minHeight: 6,
              color: color,
              backgroundColor: color.withValues(alpha: 0.15),
            ),
          ),
        ],
      ),
    );
  }

  // -------------------------------------------------------------- helpers

  Map<String, int> _totals(List<HistoryBucket> buckets) {
    final totals = <String, int>{};
    for (final b in buckets) {
      b.countsByClass.forEach((k, v) => totals[k] = (totals[k] ?? 0) + v);
    }
    return totals;
  }

  String _pct(int part, int whole) {
    if (whole == 0) return '0%';
    return '${(part * 100 / whole).round()}%';
  }

  String _two(int n) => n.toString().padLeft(2, '0');

  /// Short label for the x-axis, thinned out on dense ranges so labels
  /// don't pile up on a phone-width chart.
  String _axisLabel(List<HistoryBucket> buckets, int index) {
    final t = buckets[index].bucketStart;
    switch (_range) {
      case HistoryRange.day:
        return index % 4 == 0 ? '${_two(t.hour)}:00' : '';
      case HistoryRange.week:
        return _weekdayShort[t.weekday - 1];
      case HistoryRange.month:
        return index % 5 == 0 ? '${t.day}' : '';
      case HistoryRange.year:
        return _monthShort[t.month - 1];
    }
  }

  /// Full date/time of a bucket, for the tooltip header and the analysis.
  String _tooltipHeader(HistoryBucket b) {
    final t = b.bucketStart;
    final date = '${_monthShort[t.month - 1]} ${t.day}, ${t.year}';
    switch (_range) {
      case HistoryRange.day:
        return '$date, ${_two(t.hour)}:00-${_two(t.hour)}:59';
      case HistoryRange.week:
        return '${_weekdayShort[t.weekday - 1]}, $date';
      case HistoryRange.month:
        return date;
      case HistoryRange.year:
        return '${_monthLong[t.month - 1]} ${t.year}';
    }
  }

  String _rangeNoun() {
    switch (_range) {
      case HistoryRange.day:
        return 'today';
      case HistoryRange.week:
        return 'this week';
      case HistoryRange.month:
        return 'this month';
      case HistoryRange.year:
        return 'this year';
    }
  }

  String _bucketNoun(int n) {
    switch (_range) {
      case HistoryRange.day:
        return n == 1 ? 'hour' : 'hours';
      case HistoryRange.week:
      case HistoryRange.month:
        return n == 1 ? 'day' : 'days';
      case HistoryRange.year:
        return n == 1 ? 'month' : 'months';
    }
  }

  String _busiestLabel() {
    switch (_range) {
      case HistoryRange.day:
        return 'Busiest hour';
      case HistoryRange.week:
      case HistoryRange.month:
        return 'Busiest day';
      case HistoryRange.year:
        return 'Busiest month';
    }
  }
}
