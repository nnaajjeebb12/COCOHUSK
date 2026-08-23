import 'dart:collection';

import 'package:flutter/foundation.dart';
import 'package:logger/logger.dart';

/// One log line as captured for the in-app viewer.
class LogEntry {
  final DateTime timestamp;
  final Level level;
  final String message;
  final String? errorRepr;

  LogEntry({
    required this.timestamp,
    required this.level,
    required this.message,
    this.errorRepr,
  });

  String get levelLabel {
    switch (level) {
      case Level.trace:
        return 'TRACE';
      case Level.debug:
        return 'DEBUG';
      case Level.info:
        return 'INFO';
      case Level.warning:
        return 'WARN';
      case Level.error:
        return 'ERROR';
      case Level.fatal:
        return 'FATAL';
      default:
        return level.name.toUpperCase();
    }
  }
}

/// Centralized logger plus an in-memory ring buffer for the in-app viewer.
class AppLogger {
  AppLogger._();

  static const int _maxEntries = 500;
  static final Queue<LogEntry> _buffer = Queue<LogEntry>();
  static final ValueNotifier<int> _revision = ValueNotifier<int>(0);

  static final Logger _logger = Logger(
    filter: _ReleaseAndDebugFilter(),
    output: _BufferOutput(),
    printer: PrettyPrinter(
      methodCount: 0,
      errorMethodCount: 5,
      lineLength: 100,
      colors: false,
      printEmojis: false,
      dateTimeFormat: DateTimeFormat.onlyTimeAndSinceStart,
    ),
  );

  /// Live-updating list of buffered entries (newest at the end).
  /// Use [revision] to drive a rebuild when this changes.
  static List<LogEntry> get entries => List.unmodifiable(_buffer);

  /// Increments every time a new entry is appended.
  static ValueListenable<int> get revision => _revision;

  static void clear() {
    _buffer.clear();
    _revision.value++;
  }

  static void d(String message) => _logger.d(message);
  static void i(String message) => _logger.i(message);
  static void w(String message, [Object? error, StackTrace? stack]) =>
      _logger.w(message, error: error, stackTrace: stack);
  static void e(String message, [Object? error, StackTrace? stack]) =>
      _logger.e(message, error: error, stackTrace: stack);

  static void _append(LogEntry entry) {
    _buffer.addLast(entry);
    while (_buffer.length > _maxEntries) {
      _buffer.removeFirst();
    }
    _revision.value++;
  }
}

class _ReleaseAndDebugFilter extends LogFilter {
  @override
  bool shouldLog(LogEvent event) {
    if (kDebugMode) return true;
    return event.level.index >= Level.info.index;
  }
}

/// LogOutput that mirrors every event into [AppLogger]'s ring buffer in
/// addition to dumping its prettified lines to stdout via debugPrint.
class _BufferOutput extends LogOutput {
  @override
  void output(OutputEvent event) {
    for (final line in event.lines) {
      debugPrint(line);
    }

    final origin = event.origin;
    final msg = origin.message?.toString() ?? '';
    final errorRepr = origin.error?.toString();

    AppLogger._append(
      LogEntry(
        timestamp: origin.time,
        level: event.level,
        message: msg,
        errorRepr: errorRepr,
      ),
    );
  }
}
