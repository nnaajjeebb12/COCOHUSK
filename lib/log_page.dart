import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:logger/logger.dart';

import 'app_logger.dart';

class LogPage extends StatefulWidget {
  const LogPage({super.key});

  @override
  State<LogPage> createState() => _LogPageState();
}

class _LogPageState extends State<LogPage> {
  Level _minLevel = Level.debug;
  bool _autoScroll = true;
  final ScrollController _controller = ScrollController();

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) => _scrollToEnd());
  }

  void _scrollToEnd() {
    if (!_autoScroll || !_controller.hasClients) return;
    _controller.jumpTo(_controller.position.maxScrollExtent);
  }

  List<LogEntry> _filter(List<LogEntry> all) {
    return all.where((e) => e.level.index >= _minLevel.index).toList();
  }

  Future<void> _copyAll() async {
    final visible = _filter(AppLogger.entries);
    final text = visible.map(_formatLine).join('\n');
    await Clipboard.setData(ClipboardData(text: text));
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
          content: Text('Copied ${visible.length} log entries to clipboard.')),
    );
  }

  String _formatLine(LogEntry e) {
    final ts = e.timestamp.toIso8601String();
    final lvl = e.levelLabel.padRight(5);
    if (e.errorRepr == null) {
      return '$ts $lvl ${e.message}';
    }
    return '$ts $lvl ${e.message}  | error=${e.errorRepr}';
  }

  Color _colorFor(Level l, ColorScheme cs) {
    switch (l) {
      case Level.error:
      case Level.fatal:
        return Colors.red;
      case Level.warning:
        return Colors.orange;
      case Level.info:
        return cs.primary;
      case Level.debug:
        return cs.onSurface.withValues(alpha: 0.7);
      default:
        return cs.onSurface.withValues(alpha: 0.55);
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Logs'),
        actions: [
          PopupMenuButton<Level>(
            tooltip: 'Minimum level',
            icon: const Icon(Icons.filter_alt_outlined),
            initialValue: _minLevel,
            onSelected: (l) => setState(() => _minLevel = l),
            itemBuilder: (_) => const [
              PopupMenuItem(value: Level.debug, child: Text('Debug+')),
              PopupMenuItem(value: Level.info, child: Text('Info+')),
              PopupMenuItem(value: Level.warning, child: Text('Warn+')),
              PopupMenuItem(value: Level.error, child: Text('Error+')),
            ],
          ),
          IconButton(
            icon: Icon(_autoScroll
                ? Icons.vertical_align_bottom
                : Icons.vertical_align_center),
            tooltip: _autoScroll ? 'Auto-scroll on' : 'Auto-scroll off',
            onPressed: () => setState(() => _autoScroll = !_autoScroll),
          ),
          IconButton(
            icon: const Icon(Icons.copy_all_outlined),
            tooltip: 'Copy visible entries',
            onPressed: _copyAll,
          ),
          IconButton(
            icon: const Icon(Icons.delete_sweep_outlined),
            tooltip: 'Clear log buffer',
            onPressed: () {
              AppLogger.clear();
              setState(() {});
            },
          ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(12),
        child: Card(
          clipBehavior: Clip.antiAlias,
          child: ValueListenableBuilder<int>(
            valueListenable: AppLogger.revision,
            builder: (context, _, __) {
              final entries = _filter(AppLogger.entries);
              WidgetsBinding.instance
                  .addPostFrameCallback((_) => _scrollToEnd());

              if (entries.isEmpty) {
                return Center(
                  child: Text(
                    'No log entries yet.\nUse the app; events will appear here.',
                    textAlign: TextAlign.center,
                    style: TextStyle(
                      color: theme.colorScheme.onSurface.withValues(alpha: 0.6),
                    ),
                  ),
                );
              }

              return Scrollbar(
                controller: _controller,
                thumbVisibility: true,
                child: ListView.builder(
                  controller: _controller,
                  itemCount: entries.length,
                  itemBuilder: (context, i) {
                    final e = entries[i];
                    final ts = e.timestamp.toIso8601String().substring(11, 19);
                    return _LogTile(
                      entry: e,
                      timestamp: ts,
                      color: _colorFor(e.level, theme.colorScheme),
                    );
                  },
                ),
              );
            },
          ),
        ),
      ),
    );
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }
}

class _LogTile extends StatelessWidget {
  final LogEntry entry;
  final String timestamp;
  final Color color;

  const _LogTile({
    required this.entry,
    required this.timestamp,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 64,
            child: Text(
              timestamp,
              style: TextStyle(
                fontFamily: 'monospace',
                fontSize: 12,
                color: Theme.of(context)
                    .colorScheme
                    .onSurface
                    .withValues(alpha: 0.6),
              ),
            ),
          ),
          Container(
            margin: const EdgeInsets.only(top: 2, right: 8),
            padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 1),
            decoration: BoxDecoration(
              color: color.withValues(alpha: 0.15),
              borderRadius: BorderRadius.circular(4),
            ),
            child: Text(
              entry.levelLabel,
              style: TextStyle(
                fontFamily: 'monospace',
                fontSize: 10,
                fontWeight: FontWeight.w700,
                color: color,
              ),
            ),
          ),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                SelectableText(
                  entry.message,
                  style: const TextStyle(
                    fontFamily: 'monospace',
                    fontSize: 12,
                  ),
                ),
                if (entry.errorRepr != null) ...[
                  const SizedBox(height: 2),
                  SelectableText(
                    'error: ${entry.errorRepr}',
                    style: TextStyle(
                      fontFamily: 'monospace',
                      fontSize: 11,
                      color: Colors.red.shade400,
                    ),
                  ),
                ],
              ],
            ),
          ),
        ],
      ),
    );
  }
}
