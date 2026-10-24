import 'dart:convert';
import 'dart:io';

import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
// sqflite_common_ffi re-exports everything this file needs from sqflite
// (Database, openDatabase, databaseFactory), a separate sqflite import is
// unnecessary. The FFI backend is only actually swapped in on desktop, see
// _open() below; on Android the real sqflite plugin is still what runs.
import 'package:sqflite_common_ffi/sqflite_ffi.dart';
import 'package:uuid/uuid.dart';

import 'app_logger.dart';
import 'husk_gate.dart';
import 'inference_engine.dart';

/// One saved prediction, as shown in the History list and detail pages.
class HistoryItem {
  final int? id;
  final String imagePath;
  final PredictionResult prediction;
  final DateTime timestamp;

  HistoryItem({
    this.id,
    required this.imagePath,
    required this.prediction,
    required this.timestamp,
  });
}

/// Time-bucket granularity for the History trends chart.
enum HistoryRange { day, week, month, year }

/// One bucket's per-class counts for the trends chart, e.g. one hour (Day),
/// one calendar day (Week/Month), or one month (Year).
class HistoryBucket {
  final DateTime bucketStart;
  final Map<String, int> countsByClass;

  /// Sum of confidence values per class, so the analysis below the chart
  /// can report an average confidence without a second query.
  final Map<String, double> confidenceSumByClass;

  HistoryBucket({
    required this.bucketStart,
    required this.countsByClass,
    Map<String, double>? confidenceSumByClass,
  }) : confidenceSumByClass = confidenceSumByClass ?? {};

  int get total => countsByClass.values.fold(0, (a, b) => a + b);
}

/// A random identifier generated once per install and stamped on every
/// saved prediction.
///
/// History is local-only by design: there is no central database, no
/// server, and no sync (the app's only outbound requests are plain GETs
/// for model files, see ModelManager). Each device holds exactly one
/// device's data, its own, and nothing can read another device's rows.
///
/// device_id still earns its place: every read in [HistoryStore] filters
/// by this device's own id, so "only this device's data" is an explicit,
/// enforced rule rather than an implicit assumption, and the id is shown
/// low-key in the About dialog so a tester can say which phone a given
/// screenshot or report came from.
///
/// Deliberately not derived from a hardware identifier (Android ID, IMEI):
/// those need extra permissions and are more privacy-sensitive than
/// anything this feature needs, a locally-generated random id is enough
/// since it never leaves the device.
class DeviceId {
  static const _prefsKey = 'device_id';
  static String? _cached;

  static Future<String> get() async {
    if (_cached != null) return _cached!;
    final prefs = await SharedPreferences.getInstance();
    var id = prefs.getString(_prefsKey);
    if (id == null) {
      id = const Uuid().v4();
      await prefs.setString(_prefsKey, id);
      AppLogger.i('DeviceId: generated new device id.');
    }
    _cached = id;
    return id;
  }
}

/// Local, on-device SQLite store for prediction history. This is the
/// app's only database: there is no centralized counterpart.
///
/// Every capture path (single pick, batch, live-camera) writes here.
/// Nothing here is ever sent over the network, see [DeviceId] for what
/// device_id is and isn't doing.
class HistoryStore {
  HistoryStore._();
  static final HistoryStore instance = HistoryStore._();

  Database? _db;
  bool _ffiInitDone = false;

  Future<Database> _open() async {
    if (_db != null) return _db!;

    // sqflite only ships a native implementation for Android/iOS. Windows
    // is dev-only in this project (see the rest of the codebase's
    // Android-first / Windows-for-development split), so swap in
    // sqflite_common_ffi's desktop backend there instead.
    if (!_ffiInitDone &&
        (Platform.isWindows || Platform.isLinux || Platform.isMacOS)) {
      sqfliteFfiInit();
      databaseFactory = databaseFactoryFfi;
      _ffiInitDone = true;
    }

    final dir = await getApplicationDocumentsDirectory();
    final path = p.join(dir.path, 'husktech_history.db');

    _db = await openDatabase(
      path,
      version: 2,
      onUpgrade: (db, oldVersion, newVersion) async {
        if (oldVersion < 2) {
          // v2 records what each scan was judged against, so a history
          // entry can show its own reference values instead of whatever
          // the strictness slider reads now, plus the husk-check numbers.
          // Existing rows keep NULL and the UI omits the reference for
          // them rather than inventing thresholds they never saw.
          for (final column in const [
            'conf_threshold REAL',
            'margin_threshold REAL',
            'husk_threshold REAL',
            'husk_features TEXT',
            'rejected_by TEXT',
          ]) {
            await db.execute('ALTER TABLE predictions ADD COLUMN $column');
          }
        }
      },
      onCreate: (db, version) async {
        await db.execute('''
          CREATE TABLE predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id TEXT NOT NULL,
            timestamp_ms INTEGER NOT NULL,
            image_path TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            raw_predicted_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            margin REAL NOT NULL,
            explanation TEXT NOT NULL,
            use_cases TEXT,
            rejection_reason TEXT,
            class_probabilities TEXT NOT NULL,
            conf_threshold REAL,
            margin_threshold REAL,
            husk_threshold REAL,
            husk_features TEXT,
            rejected_by TEXT
          )
        ''');
        await db.execute(
          'CREATE INDEX idx_predictions_timestamp ON predictions(timestamp_ms)',
        );
      },
    );
    return _db!;
  }

  Future<void> insert({
    required String imagePath,
    required PredictionResult prediction,
    DateTime? timestamp,
  }) async {
    try {
      final db = await _open();
      final deviceId = await DeviceId.get();
      await db.insert('predictions', {
        'device_id': deviceId,
        'timestamp_ms': (timestamp ?? DateTime.now()).millisecondsSinceEpoch,
        'image_path': imagePath,
        'predicted_class': prediction.predictedClass,
        'raw_predicted_class': prediction.rawPredictedClass,
        'confidence': prediction.confidence,
        'margin': prediction.margin,
        'explanation': prediction.explanation,
        'use_cases': prediction.useCases,
        'rejection_reason': prediction.rejectionReason,
        'class_probabilities': jsonEncode(prediction.classProbabilities),
        'conf_threshold': prediction.confThresholdUsed,
        'margin_threshold': prediction.marginThresholdUsed,
        'husk_threshold': prediction.huskThresholdUsed,
        'husk_features': prediction.huskFeatures == null
            ? null
            : jsonEncode(prediction.huskFeatures!.toJson()),
        'rejected_by': prediction.rejectedBy.name,
      });
    } catch (e, st) {
      // A failed history write should never block or crash the actual
      // classification flow, the user already has their result on screen.
      AppLogger.e('HistoryStore: failed to save prediction.', e, st);
    }
  }

  HistoryItem _fromRow(Map<String, Object?> row) {
    final rawProbs = jsonDecode(row['class_probabilities'] as String) as Map;
    final classProbabilities = rawProbs.map(
      (k, v) => MapEntry(k as String, (v as num).toDouble()),
    );
    return HistoryItem(
      id: row['id'] as int,
      timestamp:
          DateTime.fromMillisecondsSinceEpoch(row['timestamp_ms'] as int),
      imagePath: row['image_path'] as String,
      prediction: PredictionResult(
        predictedClass: row['predicted_class'] as String,
        confidence: row['confidence'] as double,
        explanation: row['explanation'] as String,
        useCases: row['use_cases'] as String?,
        classProbabilities: classProbabilities,
        rawPredictedClass: row['raw_predicted_class'] as String,
        margin: row['margin'] as double,
        rejectionReason: row['rejection_reason'] as String?,
        huskFeatures: HuskFeatures.fromJson(row['husk_features'] as String?),
        // Null for rows written before schema v2; PredictionResult treats a
        // zero threshold as "no reference recorded" and the UI hides the
        // reference block rather than showing 0%.
        confThresholdUsed: (row['conf_threshold'] as num?)?.toDouble() ?? 0.0,
        marginThresholdUsed:
            (row['margin_threshold'] as num?)?.toDouble() ?? 0.0,
        huskThresholdUsed: (row['husk_threshold'] as num?)?.toDouble() ?? 0.0,
        rejectedBy: _rejectionCheckFrom(row['rejected_by'] as String?),
      ),
    );
  }

  static RejectionCheck _rejectionCheckFrom(String? name) {
    if (name == null) return RejectionCheck.none;
    for (final value in RejectionCheck.values) {
      if (value.name == name) return value;
    }
    return RejectionCheck.none;
  }

  /// All of this device's history, most recent first.
  Future<List<HistoryItem>> fetchAll({int? limit}) async {
    try {
      final db = await _open();
      final deviceId = await DeviceId.get();
      final rows = await db.query(
        'predictions',
        where: 'device_id = ?',
        whereArgs: [deviceId],
        orderBy: 'timestamp_ms DESC',
        limit: limit,
      );
      return rows.map(_fromRow).toList();
    } catch (e, st) {
      AppLogger.e('HistoryStore: failed to load history.', e, st);
      return [];
    }
  }

  /// Buckets this device's predictions into [range]-sized buckets ending
  /// now, counting how many of each class fell into each bucket. Buckets
  /// are returned in chronological order, including empty ones, so the
  /// chart's x-axis is stable even with no data in a given slot.
  Future<List<HistoryBucket>> aggregate(HistoryRange range,
      {DateTime? now}) async {
    final effectiveNow = now ?? DateTime.now();
    final start = _rangeStart(range, effectiveNow);
    final buckets = _emptyBuckets(range, start);

    try {
      final db = await _open();
      final deviceId = await DeviceId.get();
      final rows = await db.query(
        'predictions',
        columns: ['timestamp_ms', 'predicted_class', 'confidence'],
        where: 'device_id = ? AND timestamp_ms >= ?',
        whereArgs: [deviceId, start.millisecondsSinceEpoch],
      );

      for (final row in rows) {
        final ts =
            DateTime.fromMillisecondsSinceEpoch(row['timestamp_ms'] as int);
        final cls = row['predicted_class'] as String;
        final conf = (row['confidence'] as num).toDouble();
        final bucket = buckets[_bucketKey(range, ts)];
        if (bucket != null) {
          bucket.countsByClass[cls] = (bucket.countsByClass[cls] ?? 0) + 1;
          bucket.confidenceSumByClass[cls] =
              (bucket.confidenceSumByClass[cls] ?? 0) + conf;
        }
      }
    } catch (e, st) {
      AppLogger.e('HistoryStore: failed to aggregate history.', e, st);
    }

    return buckets.values.toList();
  }

  DateTime _rangeStart(HistoryRange range, DateTime now) {
    switch (range) {
      case HistoryRange.day:
        return DateTime(now.year, now.month, now.day);
      case HistoryRange.week:
        final startOfToday = DateTime(now.year, now.month, now.day);
        return startOfToday
            .subtract(Duration(days: startOfToday.weekday - 1));
      case HistoryRange.month:
        return DateTime(now.year, now.month, 1);
      case HistoryRange.year:
        return DateTime(now.year, 1, 1);
    }
  }

  String _bucketKey(HistoryRange range, DateTime ts) {
    switch (range) {
      case HistoryRange.day:
        return '${ts.year}-${ts.month}-${ts.day}-${ts.hour}';
      case HistoryRange.week:
      case HistoryRange.month:
        return '${ts.year}-${ts.month}-${ts.day}';
      case HistoryRange.year:
        return '${ts.year}-${ts.month}';
    }
  }

  Map<String, HistoryBucket> _emptyBuckets(HistoryRange range,
      DateTime start) {
    final map = <String, HistoryBucket>{};
    switch (range) {
      case HistoryRange.day:
        for (int h = 0; h < 24; h++) {
          final t = DateTime(start.year, start.month, start.day, h);
          map[_bucketKey(range, t)] =
              HistoryBucket(bucketStart: t, countsByClass: {});
        }
        break;
      case HistoryRange.week:
        for (int d = 0; d < 7; d++) {
          final t = start.add(Duration(days: d));
          map[_bucketKey(range, t)] =
              HistoryBucket(bucketStart: t, countsByClass: {});
        }
        break;
      case HistoryRange.month:
        final daysInMonth = DateTime(start.year, start.month + 1, 0).day;
        for (int d = 0; d < daysInMonth; d++) {
          final t = start.add(Duration(days: d));
          map[_bucketKey(range, t)] =
              HistoryBucket(bucketStart: t, countsByClass: {});
        }
        break;
      case HistoryRange.year:
        for (int m = 1; m <= 12; m++) {
          final t = DateTime(start.year, m, 1);
          map[_bucketKey(range, t)] =
              HistoryBucket(bucketStart: t, countsByClass: {});
        }
        break;
    }
    return map;
  }
}
