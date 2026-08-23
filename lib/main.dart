import 'dart:io';
import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';

import 'app_logger.dart';
import 'inference_engine.dart';
import 'live_camera_page.dart';
import 'log_page.dart';
import 'model_manager.dart';

bool get _platformSupportsLiveCamera {
  if (kIsWeb) return false;
  return Platform.isAndroid || Platform.isIOS;
}

bool get _platformSupportsImagePickerCamera {
  if (kIsWeb) return false;
  return Platform.isAndroid || Platform.isIOS;
}

/// Shared "immature" -> "Immature" style formatting for a predicted class,
/// used anywhere a result is shown.
String formatClassName(String className) {
  if (className.isEmpty) return className;
  return className
      .split('_')
      .map((p) =>
          p.isEmpty ? p : p[0].toUpperCase() + p.substring(1).toLowerCase())
      .join(' ');
}

/// Shared badge color for a predicted class, used anywhere a result is
/// shown: the home page result card, batch results, and history.
Color classBadgeColor(String predictedClass, Color fallback) {
  switch (predictedClass) {
    case 'immature':
      return Colors.orange;
    case 'mature':
      return Colors.green;
    case 'overmature':
      return Colors.brown;
    case 'rejected':
      return Colors.red;
    default:
      return fallback;
  }
}

/// Shared icon for a predicted class, paired with [classBadgeColor].
IconData classIcon(String predictedClass) {
  switch (predictedClass) {
    case 'immature':
      return Icons.hourglass_top;
    case 'mature':
      return Icons.star;
    case 'overmature':
      return Icons.hourglass_bottom;
    case 'rejected':
      return Icons.block;
    default:
      return Icons.help_outline;
  }
}

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  AppLogger.i('HuskTech starting up.');
  runApp(const HuskTechApp());
}

class HuskTechApp extends StatefulWidget {
  const HuskTechApp({super.key});

  @override
  State<HuskTechApp> createState() => _HuskTechAppState();
}

class _HuskTechAppState extends State<HuskTechApp> {
  ThemeMode _themeMode = ThemeMode.light;

  void _toggleTheme(bool isDark) {
    setState(() => _themeMode = isDark ? ThemeMode.dark : ThemeMode.light);
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'HuskTech',
      debugShowCheckedModeBanner: false,
      themeMode: _themeMode,
      theme: _buildLightTheme(),
      darkTheme: _buildDarkTheme(),
      home: HomePage(themeMode: _themeMode, onThemeChanged: _toggleTheme),
    );
  }

  ThemeData _buildLightTheme() {
    const seedColor = Color(0xFF00695C);
    return ThemeData(
      colorScheme: ColorScheme.fromSeed(seedColor: seedColor),
      useMaterial3: true,
      scaffoldBackgroundColor: const Color(0xFFF5F5F5),
      cardTheme: CardThemeData(
        elevation: 4,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
        ),
      ),
      appBarTheme: const AppBarTheme(centerTitle: true),
    );
  }

  ThemeData _buildDarkTheme() {
    const seedColor = Color(0xFF80CBC4);
    return ThemeData(
      colorScheme: ColorScheme.fromSeed(
        seedColor: seedColor,
        brightness: Brightness.dark,
      ),
      useMaterial3: true,
      scaffoldBackgroundColor: const Color(0xFF121212),
      cardTheme: CardThemeData(
        elevation: 3,
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
        ),
      ),
      appBarTheme: const AppBarTheme(centerTitle: true),
    );
  }
}

class HistoryItem {
  final String imagePath;
  final PredictionResult prediction;
  final DateTime timestamp;

  HistoryItem({
    required this.imagePath,
    required this.prediction,
    required this.timestamp,
  });
}

class BatchItem {
  final String imagePath;
  final String predictedClass;
  final double confidence;

  BatchItem({
    required this.imagePath,
    required this.predictedClass,
    required this.confidence,
  });
}

class HomePage extends StatefulWidget {
  final ThemeMode themeMode;
  final ValueChanged<bool> onThemeChanged;

  const HomePage({
    super.key,
    required this.themeMode,
    required this.onThemeChanged,
  });

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final InferenceEngine _engine = InferenceEngine();
  final ImagePicker _picker = ImagePicker();

  File? _selectedImage;
  bool _isLoading = false;
  bool _isModelLoaded = false;
  String? _errorMessage;
  PredictionResult? _prediction;

  final List<HistoryItem> _history = [];
  final List<BatchItem> _batchResults = [];

  @override
  void initState() {
    super.initState();
    _loadEngine();
  }

  Future<void> _loadEngine() async {
    AppLogger.i('Engine load: start');
    try {
      await _engine.loadDefaultModel();
      if (!mounted) return;
      setState(() {
        _isModelLoaded = true;
        _errorMessage = null;
      });
      AppLogger.i('Engine load: success');
    } catch (e, st) {
      AppLogger.e('Engine load: failure', e, st);
      if (!mounted) return;
      setState(() {
        _isModelLoaded = false;
        _errorMessage = 'Failed to load model. Please restart the app.';
      });
    }
  }

  Future<void> _loadCustomModel() async {
    AppLogger.i('Custom model: user initiated picker.');
    setState(() => _isLoading = true);
    try {
      // Use FileType.any rather than FileType.custom: Android's Storage
      // Access Framework has no registered MIME type for ".tflite", so the
      // picker rejects allowedExtensions with "Unsupported filter". We let
      // the user pick any file and validate the extension + interpret-
      // ability ourselves. withData: true makes the picker return raw bytes
      // through the platform channel, which sidesteps content:// URIs we
      // cannot open as plain File paths.
      final result = await FilePicker.platform.pickFiles(
        type: FileType.any,
        withData: true,
      );

      if (result == null || result.files.isEmpty) {
        AppLogger.i('Custom model: picker cancelled.');
        return;
      }

      final picked = result.files.single;
      final fileName = picked.name;

      if (!fileName.toLowerCase().endsWith('.tflite')) {
        AppLogger.w('Custom model: rejected non-.tflite file "$fileName"');
        _showSnack(
          'Selected file is not a .tflite model: $fileName',
          isError: true,
        );
        return;
      }

      Uint8List? bytes = picked.bytes;
      if (bytes == null && picked.path != null) {
        // Fallback for platforms where file_picker can return a real path
        // but not the bytes (e.g. desktop).
        AppLogger.d('Custom model: reading bytes from path ${picked.path}');
        bytes = await File(picked.path!).readAsBytes();
      }

      if (bytes == null) {
        AppLogger.w('Custom model: picker returned neither bytes nor path.');
        _showSnack('Could not read the selected file.', isError: true);
        return;
      }

      AppLogger.i('Custom model: picked "$fileName" '
          '(${bytes.lengthInBytes} bytes)');

      try {
        await _engine.loadCustomFromBytes(
          bytes: bytes,
          displayName: fileName,
        );
      } catch (e, st) {
        AppLogger.e('Custom model: file is not a valid TFLite model.', e, st);
        _showSnack(
          'Failed to load model: not a valid .tflite file.\n'
          'Details: ${_short(e)}',
          isError: true,
        );
        return;
      }

      if (!mounted) return;
      setState(() {
        _isModelLoaded = true;
        _errorMessage = null;
        _prediction = null;
        _selectedImage = null;
      });
      _showSnack('New model loaded: $fileName');
    } catch (e, st) {
      AppLogger.e('Custom model: unexpected error during load.', e, st);
      if (!mounted) return;
      _showSnack(
        'Failed to load custom model. Reverting to previous.\n'
        'Details: ${_short(e)}',
        isError: true,
      );
      await _loadEngine();
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  String _short(Object e) {
    final s = e.toString();
    return s.length <= 140 ? s : '${s.substring(0, 140)}...';
  }

  Future<void> _resetToDefaultModel() async {
    AppLogger.i('Custom model: reverting to default.');
    await ModelManager.clearCustomModelPath();
    await _loadEngine();
    if (!mounted) return;
    _showSnack('Reverted to default model.');
  }

  /// Re-runs the same load path as app startup: re-checks for a cloud
  /// update, then re-reads whichever model is currently active (custom
  /// path takes priority, else the cached/cloud model, else bundled) and
  /// rebuilds the interpreter from scratch. Lets the user recover from a
  /// failed load, or pick up a custom model file that changed on disk,
  /// without restarting the app.
  Future<void> _reloadModel() async {
    if (_isLoading) return;
    AppLogger.i('Model reload: user requested.');
    setState(() => _isLoading = true);
    try {
      await _loadEngine();
      if (!mounted) return;
      if (_isModelLoaded && _errorMessage == null) {
        final warning = _engine.verificationWarning;
        _showSnack(
          warning == null
              ? 'Model reloaded: ${_engine.activeName ?? "default"}.'
              : 'Model reloaded, but: $warning',
          isError: warning != null,
        );
      }
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  void _showSnack(String message, {bool isError = false}) {
    if (!mounted) return;
    final messenger = ScaffoldMessenger.of(context);
    messenger.hideCurrentSnackBar();
    messenger.showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: isError ? Colors.red.shade700 : null,
        behavior: SnackBarBehavior.floating,
        duration: const Duration(seconds: 4),
      ),
    );
  }

  Future<void> _openLiveCamera() async {
    if (!_engine.isReady) {
      setState(() => _errorMessage = 'Model not loaded yet. Please wait.');
      return;
    }
    final result = await Navigator.of(context).push<CapturedFrameResult>(
      MaterialPageRoute(builder: (_) => LiveCameraPage(engine: _engine)),
    );
    if (result == null) return;
    await _persistCapturedFrame(result);
  }

  Future<void> _persistCapturedFrame(CapturedFrameResult captured) async {
    try {
      final dir = await getApplicationDocumentsDirectory();
      final path =
          '${dir.path}/capture_${DateTime.now().millisecondsSinceEpoch}.jpg';
      final f = await File(path).writeAsBytes(captured.bytes, flush: true);
      _history.add(
        HistoryItem(
          imagePath: f.path,
          prediction: captured.prediction,
          timestamp: DateTime.now(),
        ),
      );
      if (!mounted) return;
      setState(() {
        _selectedImage = f;
        _prediction = captured.prediction;
      });
    } catch (e, st) {
      AppLogger.e('Failed to persist captured frame.', e, st);
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    if (!_engine.isReady) {
      setState(() => _errorMessage = 'Model not loaded yet. Please wait.');
      return;
    }
    if (source == ImageSource.camera && !_platformSupportsImagePickerCamera) {
      _showSnack('Camera is not available on this platform.', isError: true);
      return;
    }

    setState(() {
      _errorMessage = null;
      _prediction = null;
    });

    try {
      final picked = await _picker.pickImage(source: source, imageQuality: 90);
      if (picked == null) return;
      setState(() => _selectedImage = File(picked.path));
      await _runInference();
    } catch (e, st) {
      AppLogger.e('Error picking image from ${source.name}.', e, st);
      if (!mounted) return;
      setState(() => _errorMessage = 'Error picking image. Please try again.');
    }
  }

  Future<void> _pickBatchFromGallery() async {
    if (!_engine.isReady) {
      setState(() => _errorMessage = 'Model not loaded yet. Please wait.');
      return;
    }
    setState(() {
      _errorMessage = null;
      _prediction = null;
      _isLoading = true;
      _batchResults.clear();
    });

    try {
      final picked = await _picker.pickMultiImage(imageQuality: 90);
      if (picked.isEmpty) {
        setState(() => _isLoading = false);
        return;
      }

      setState(() => _selectedImage = File(picked.first.path));

      final results = <BatchItem>[];
      for (final xfile in picked) {
        final f = File(xfile.path);
        final p = await _engine.predictFromFile(f);
        if (p != null) {
          results.add(BatchItem(
            imagePath: f.path,
            predictedClass: p.predictedClass,
            confidence: p.confidence,
          ));
          _history.add(HistoryItem(
            imagePath: f.path,
            prediction: p,
            timestamp: DateTime.now(),
          ));
        }
      }

      setState(() {
        _batchResults
          ..clear()
          ..addAll(results);
        _isLoading = false;
      });

      if (!mounted) return;
      Navigator.of(context).push(
        MaterialPageRoute(
          builder: (_) => BatchPage(batchResults: _batchResults),
        ),
      );
    } catch (e, st) {
      AppLogger.e('Batch analysis error.', e, st);
      if (!mounted) return;
      setState(() {
        _errorMessage = 'Error during batch analysis. Please try again.';
        _isLoading = false;
      });
    }
  }

  Future<void> _runInference() async {
    if (_selectedImage == null) return;
    if (!_engine.isReady) {
      setState(() => _errorMessage = 'Model not ready. Please wait.');
      return;
    }
    setState(() {
      _isLoading = true;
      _prediction = null;
      _errorMessage = null;
    });

    final result = await _engine.predictFromFile(_selectedImage!);
    if (result == null) {
      setState(() {
        _errorMessage = 'Failed to analyze image.';
        _isLoading = false;
      });
      return;
    }

    _history.add(HistoryItem(
      imagePath: _selectedImage!.path,
      prediction: result,
      timestamp: DateTime.now(),
    ));

    setState(() {
      _prediction = result;
      _isLoading = false;
    });
  }

  @override
  void dispose() {
    _engine.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final ready = _isModelLoaded && _errorMessage == null;
    final isDark = widget.themeMode == ThemeMode.dark;
    final colorScheme = Theme.of(context).colorScheme;

    return Scaffold(
      appBar: AppBar(
        title: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            ClipRRect(
              borderRadius: BorderRadius.circular(6),
              child: Image.asset(
                'huskpng.png',
                width: 28,
                height: 28,
                errorBuilder: (_, __, ___) =>
                    const SizedBox(width: 28, height: 28),
              ),
            ),
            const SizedBox(width: 10),
            const Flexible(
              child: Text(
                'HuskTech: Coconut Husk Grading',
                overflow: TextOverflow.ellipsis,
              ),
            ),
          ],
        ),
        actions: [
          IconButton(
            icon: _isLoading
                ? const SizedBox(
                    width: 20,
                    height: 20,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Icon(Icons.refresh),
            tooltip: 'Reload model',
            onPressed: _isLoading ? null : _reloadModel,
          ),
          IconButton(
            icon: const Icon(Icons.history),
            tooltip: 'History',
            onPressed: () {
              Navigator.of(context).push(
                MaterialPageRoute(
                  builder: (_) => HistoryPage(history: _history),
                ),
              );
            },
          ),
          PopupMenuButton<String>(
            onSelected: (value) async {
              switch (value) {
                case 'info':
                  Navigator.of(context).push(
                    MaterialPageRoute(builder: (_) => const InfoPage()),
                  );
                  break;
                case 'manual':
                  Navigator.of(context).push(
                    MaterialPageRoute(builder: (_) => const ManualPage()),
                  );
                  break;
                case 'model':
                  await Navigator.of(context).push(
                    MaterialPageRoute(
                      builder: (_) => ModelSettingsPage(
                        engine: _engine,
                        onPickCustom: _loadCustomModel,
                        onResetDefault: _resetToDefaultModel,
                        onReload: _reloadModel,
                      ),
                    ),
                  );
                  setState(() {}); // refresh header chips with new model info
                  break;
                case 'model_guide':
                  Navigator.of(context).push(
                    MaterialPageRoute(
                      builder: (_) => const ModelUploadGuidePage(),
                    ),
                  );
                  break;
                case 'logs':
                  Navigator.of(context).push(
                    MaterialPageRoute(builder: (_) => const LogPage()),
                  );
                  break;
                case 'about':
                  showAboutDialog(
                    context: context,
                    applicationName: 'HuskTech',
                    applicationVersion: '1.0.0',
                    applicationIcon: ClipRRect(
                      borderRadius: BorderRadius.circular(12),
                      child: Image.asset(
                        'huskpng.png',
                        width: 48,
                        height: 48,
                        errorBuilder: (_, __, ___) =>
                            const SizedBox(width: 48, height: 48),
                      ),
                    ),
                    children: const [
                      Text(
                        'HuskTech is an offline coconut husk maturity classification app.\n\n'
                        'It runs entirely on-device using a TensorFlow Lite model '
                        'with dynamic model updates from the cloud or a user-supplied file.',
                      ),
                    ],
                  );
                  break;
              }
            },
            itemBuilder: (context) => const [
              PopupMenuItem(value: 'info', child: Text('View Information')),
              PopupMenuItem(value: 'manual', child: Text('Read Manual')),
              PopupMenuItem(value: 'model', child: Text('Model Settings')),
              PopupMenuItem(
                  value: 'model_guide', child: Text('Custom Model Guide')),
              PopupMenuItem(value: 'logs', child: Text('View Logs')),
              PopupMenuItem(value: 'about', child: Text('About')),
            ],
          ),
          Row(
            children: [
              Icon(Icons.light_mode,
                  color: isDark ? Colors.grey : colorScheme.primary, size: 20),
              Switch.adaptive(
                value: isDark,
                onChanged: widget.onThemeChanged,
                activeTrackColor: colorScheme.primary,
              ),
              Icon(Icons.dark_mode,
                  color: isDark ? colorScheme.primary : Colors.grey, size: 20),
              const SizedBox(width: 4),
            ],
          ),
        ],
      ),
      body: SafeArea(
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 12, 16, 4),
              child: _buildHeader(context, ready),
            ),
            const SizedBox(height: 4),
            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.fromLTRB(16, 8, 16, 8),
                child: Column(
                  children: [
                    _buildImageSection(context, ready),
                    const SizedBox(height: 16),
                    _buildResultCard(context),
                    const SizedBox(height: 16),
                    _buildMaturityChips(context),
                  ],
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 8, 16, 4),
              child: Row(
                children: [
                  if (_platformSupportsLiveCamera) ...[
                    Expanded(
                      child: FilledButton.icon(
                        onPressed:
                            (!ready || _isLoading) ? null : _openLiveCamera,
                        icon: const Icon(Icons.videocam),
                        label: const Text('Live Camera'),
                      ),
                    ),
                    const SizedBox(width: 12),
                  ],
                  Expanded(
                    child: _platformSupportsLiveCamera
                        ? OutlinedButton.icon(
                            onPressed: (!ready || _isLoading)
                                ? null
                                : () => _pickImage(ImageSource.gallery),
                            icon: const Icon(Icons.photo_library),
                            label: const Text('Gallery'),
                          )
                        : FilledButton.icon(
                            onPressed: (!ready || _isLoading)
                                ? null
                                : () => _pickImage(ImageSource.gallery),
                            icon: const Icon(Icons.photo_library),
                            label: const Text('Gallery'),
                          ),
                  ),
                ],
              ),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 4, 16, 16),
              child: Row(
                children: [
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed:
                          (!ready || _isLoading) ? null : _pickBatchFromGallery,
                      icon: const Icon(Icons.collections),
                      label: const Text('Batch (Gallery)'),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: _isLoading ? null : _loadCustomModel,
                      icon: const Icon(Icons.upload_file),
                      label: const Text('Load Model'),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildHeader(BuildContext context, bool ready) {
    final textTheme = Theme.of(context).textTheme;
    final colorScheme = Theme.of(context).colorScheme;

    String sourceLabel;
    switch (_engine.source) {
      case ModelSource.cloud:
        sourceLabel = 'Cloud-updated';
        break;
      case ModelSource.local:
        sourceLabel = 'Local';
        break;
      case ModelSource.custom:
        sourceLabel = 'Custom';
        break;
      case ModelSource.bundled:
        sourceLabel = 'Bundled';
        break;
    }

    String versionText;
    if (_engine.source == ModelSource.custom) {
      versionText = 'Model: ${_engine.activeName ?? "custom"} • $sourceLabel';
    } else if (_engine.modelVersion > 0) {
      versionText = 'Model version: ${_engine.modelVersion} • $sourceLabel';
    } else {
      versionText = 'Model version: not recorded • $sourceLabel';
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Coconut Husk Maturity Classification',
          style: textTheme.titleLarge?.copyWith(
            fontWeight: FontWeight.bold,
            color: colorScheme.primary,
          ),
        ),
        const SizedBox(height: 4),
        Text(
          ready
              ? 'Use Live Camera for continuous prediction, Gallery to pick a saved image, or Batch to analyze many at once.'
              : 'Loading on-device model. Please wait...',
          style: textTheme.bodyMedium?.copyWith(
            color: colorScheme.onSurface.withValues(alpha: 0.7),
          ),
        ),
        const SizedBox(height: 4),
        Text(
          versionText,
          style: textTheme.bodySmall?.copyWith(
            color: colorScheme.onSurface.withValues(alpha: 0.6),
          ),
        ),
      ],
    );
  }

  Widget _buildImageSection(BuildContext context, bool ready) {
    final textColor =
        Theme.of(context).colorScheme.onSurface.withValues(alpha: 0.7);

    if (_selectedImage == null) {
      return Card(
        child: SizedBox(
          width: double.infinity,
          height: 260,
          child: Center(
            child: Text(
              ready
                  ? (_platformSupportsLiveCamera
                      ? 'No image selected.\nUse Live Camera, Gallery, or Batch below.'
                      : 'No image selected.\nUse the Gallery button below.')
                  : 'Preparing model...',
              textAlign: TextAlign.center,
              style: TextStyle(color: textColor),
            ),
          ),
        ),
      );
    }

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          children: [
            ClipRRect(
              borderRadius: BorderRadius.circular(12),
              child: Image.file(
                _selectedImage!,
                fit: BoxFit.cover,
                width: double.infinity,
                height: 260,
                errorBuilder: (_, __, ___) =>
                    const Icon(Icons.broken_image, size: 48),
              ),
            ),
            const SizedBox(height: 12),
            if (_isLoading) const CircularProgressIndicator(),
          ],
        ),
      ),
    );
  }

  Widget _buildResultCard(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: _buildResultSection(colorScheme, context),
      ),
    );
  }

  Widget _buildResultSection(ColorScheme colorScheme, BuildContext context) {
    if (_errorMessage != null) {
      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(_errorMessage!, style: const TextStyle(color: Colors.red)),
          const SizedBox(height: 12),
          OutlinedButton.icon(
            onPressed: _isLoading ? null : _reloadModel,
            icon: const Icon(Icons.refresh),
            label: const Text('Retry loading model'),
          ),
        ],
      );
    }
    if (!_isModelLoaded) {
      return const Text('Loading model...',
          style: TextStyle(fontStyle: FontStyle.italic));
    }
    if (_isLoading) {
      return const Text('Analyzing image...',
          style: TextStyle(fontStyle: FontStyle.italic));
    }
    if (_prediction == null) {
      return Text(
        'Result will appear here after analysis.',
        style: TextStyle(color: colorScheme.onSurface.withValues(alpha: 0.6)),
      );
    }

    final p = _prediction!;
    final confidencePercent = (p.confidence * 100).toStringAsFixed(1);
    final readableClass = formatClassName(p.predictedClass);
    final badgeColor = classBadgeColor(p.predictedClass, colorScheme.primary);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text('Prediction',
            style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
        const SizedBox(height: 8),
        Row(
          children: [
            Container(
              padding: const EdgeInsets.symmetric(vertical: 4, horizontal: 10),
              decoration: BoxDecoration(
                color: badgeColor.withValues(alpha: 0.15),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Text(
                'Stage: $readableClass',
                style: TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                  color: badgeColor,
                ),
              ),
            ),
            const SizedBox(width: 12),
            Text('Confidence: $confidencePercent%',
                style: const TextStyle(fontSize: 14)),
          ],
        ),
        const SizedBox(height: 8),
        Text(p.explanation, style: const TextStyle(fontSize: 13)),
      ],
    );
  }

  Widget _buildMaturityChips(BuildContext context) {
    return Align(
      alignment: Alignment.centerLeft,
      child: Wrap(
        spacing: 8,
        runSpacing: 8,
        children: [
          Chip(
            label: const Text('Immature'),
            avatar: const Icon(Icons.hourglass_top, size: 18),
            backgroundColor: Colors.orange.withValues(alpha: 0.20),
          ),
          Chip(
            label: const Text('Mature'),
            avatar: const Icon(Icons.star, size: 18),
            backgroundColor: Colors.green.withValues(alpha: 0.20),
          ),
          Chip(
            label: const Text('Overmature'),
            avatar: const Icon(Icons.hourglass_bottom, size: 18),
            backgroundColor: Colors.brown.withValues(alpha: 0.20),
          ),
          Chip(
            label: const Text('Rejected'),
            avatar: const Icon(Icons.block, size: 18),
            backgroundColor: Colors.red.withValues(alpha: 0.20),
          ),
        ],
      ),
    );
  }
}

/// ===== MODEL SETTINGS PAGE =====
///
/// Reads live off [engine] (rather than a snapshot of its fields) so that
/// after a pick/reset/reload the card and verification status update in
/// place without needing to pop and re-push this route.
class ModelSettingsPage extends StatefulWidget {
  final InferenceEngine engine;
  final Future<void> Function() onPickCustom;
  final Future<void> Function() onResetDefault;
  final Future<void> Function() onReload;

  const ModelSettingsPage({
    super.key,
    required this.engine,
    required this.onPickCustom,
    required this.onResetDefault,
    required this.onReload,
  });

  @override
  State<ModelSettingsPage> createState() => _ModelSettingsPageState();
}

class _ModelSettingsPageState extends State<ModelSettingsPage> {
  bool _busy = false;

  Future<void> _run(Future<void> Function() action) async {
    if (_busy) return;
    setState(() => _busy = true);
    try {
      await action();
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final engine = widget.engine;
    final source = engine.source;
    final verified = engine.isVerified;
    final warning = engine.verificationWarning;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Model Settings'),
        actions: [
          IconButton(
            icon: _busy
                ? const SizedBox(
                    width: 20,
                    height: 20,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Icon(Icons.refresh),
            tooltip: 'Reload current model',
            onPressed: _busy ? null : () => _run(widget.onReload),
          ),
          IconButton(
            icon: const Icon(Icons.help_outline),
            tooltip: 'Custom Model Guide',
            onPressed: () => Navigator.of(context).push(
              MaterialPageRoute(builder: (_) => const ModelUploadGuidePage()),
            ),
          ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Active model', style: theme.textTheme.titleMedium),
            const SizedBox(height: 8),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Icon(
                          verified
                              ? Icons.check_circle
                              : Icons.warning_amber_rounded,
                          size: 18,
                          color: verified ? Colors.green : Colors.orange,
                        ),
                        const SizedBox(width: 6),
                        Expanded(
                          child: Text(
                            verified
                                ? 'Verified: loaded and matches this app\'s '
                                    'expected input/output.'
                                : (warning ?? 'Model not loaded.'),
                            style: theme.textTheme.bodySmall?.copyWith(
                              color: verified
                                  ? Colors.green.shade700
                                  : Colors.orange.shade800,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ),
                      ],
                    ),
                    const Divider(height: 20),
                    Text('Name: ${engine.activeName ?? "(bundled default)"}'),
                    const SizedBox(height: 4),
                    Text('Source: ${source.name}'),
                    if (engine.modelVersion > 0) ...[
                      const SizedBox(height: 4),
                      Text('Version: ${engine.modelVersion}'),
                    ],
                    if (engine.activePath != null) ...[
                      const SizedBox(height: 4),
                      Text('Path: ${engine.activePath}',
                          style: theme.textTheme.bodySmall),
                    ],
                    if (engine.inputShape != null ||
                        engine.outputShape != null) ...[
                      const SizedBox(height: 4),
                      Text(
                        'Tensors: input ${engine.inputShape ?? "-"} → '
                        'output ${engine.outputShape ?? "-"}',
                        style: theme.textTheme.bodySmall,
                      ),
                    ],
                    if (engine.labels.isNotEmpty) ...[
                      const SizedBox(height: 8),
                      Text('Classes (in output order):',
                          style: theme.textTheme.bodySmall
                              ?.copyWith(fontWeight: FontWeight.w600)),
                      const SizedBox(height: 4),
                      Wrap(
                        spacing: 6,
                        runSpacing: 6,
                        children: [
                          for (final label in engine.labels)
                            Chip(
                              label: Text(label),
                              visualDensity: VisualDensity.compact,
                            ),
                        ],
                      ),
                    ],
                  ],
                ),
              ),
            ),
            const SizedBox(height: 24),
            FilledButton.icon(
              onPressed: _busy ? null : () => _run(widget.onPickCustom),
              icon: const Icon(Icons.upload_file),
              label: const Text('Load a new .tflite model'),
            ),
            const SizedBox(height: 12),
            OutlinedButton.icon(
              onPressed: _busy
                  ? null
                  : (source == ModelSource.custom
                      ? () => _run(widget.onResetDefault)
                      : null),
              icon: const Icon(Icons.restore),
              label: const Text('Revert to default model'),
            ),
            const SizedBox(height: 12),
            OutlinedButton.icon(
              onPressed: _busy ? null : () => _run(widget.onReload),
              icon: _busy
                  ? const SizedBox(
                      width: 16,
                      height: 16,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(Icons.refresh),
              label: Text(_busy ? 'Reloading…' : 'Reload current model'),
            ),
            const SizedBox(height: 12),
            TextButton.icon(
              onPressed: () => Navigator.of(context).push(
                MaterialPageRoute(builder: (_) => const ModelUploadGuidePage()),
              ),
              icon: const Icon(Icons.menu_book),
              label: const Text('What model can I upload?'),
            ),
            const Spacer(),
            Text(
              'The selected custom model is copied into app storage and reloaded '
              'automatically on next launch. If a custom model fails to load, '
              'the app falls back to the default bundled model.',
              style: theme.textTheme.bodySmall,
            ),
          ],
        ),
      ),
    );
  }
}

/// ===== CUSTOM MODEL UPLOAD GUIDE =====
class ModelUploadGuidePage extends StatelessWidget {
  const ModelUploadGuidePage({super.key});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final body = theme.textTheme.bodyMedium;
    final small = theme.textTheme.bodySmall;
    final h =
        theme.textTheme.titleMedium?.copyWith(fontWeight: FontWeight.w700);

    return Scaffold(
      appBar: AppBar(title: const Text('Custom Model Guide')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Text('What HuskTech expects from your model', style: h),
          const SizedBox(height: 8),
          Text(
            'You can swap the bundled coconut-husk classifier for any '
            'image-classification model, for example a retrained model with '
            'more grades, or a regional variant. As long as the file matches '
            'the contract below, the app will load it without recompiling.',
            style: body,
          ),
          const SizedBox(height: 16),
          _section(
            theme,
            icon: Icons.description_outlined,
            title: 'File format',
            children: const [
              'TensorFlow Lite: a single `.tflite` file.',
              'Quantized (INT8) or float32; both work.',
              'Typical size: 2-50 MB.',
            ],
          ),
          _section(
            theme,
            icon: Icons.input,
            title: 'Input tensor',
            children: const [
              'Shape: [1, 224, 224, 3]',
              'dtype: float32',
              'Pixel range: 0-255 raw (the app does NOT normalize pixels '
                  'itself: bake any rescaling/normalization into the model '
                  'as its own first layer, e.g. Keras `Rescaling(1/255)` or '
                  'a `preprocess_input` that expects raw pixel values).',
              'Color layout: RGB (not BGR). Alpha channel is dropped.',
            ],
          ),
          _section(
            theme,
            icon: Icons.output,
            title: 'Output tensor',
            children: const [
              'Shape: [1, N] where N = number of classes.',
              'dtype: float32, treated as a probability distribution.',
              'If outputs do not sum to 1, the app normalizes before applying '
                  'the rejection layer.',
            ],
          ),
          _section(
            theme,
            icon: Icons.label_outline,
            title: 'Class labels',
            children: const [
              'The app currently uses the labels in '
                  '`assets/model/class_names.txt`, one label per line, in the '
                  'same order as the model output channels.',
              'For best results, train against the same label set in the same '
                  'order: immature, mature, overmature.',
              'A "rejected" output is produced by the app itself when '
                  'confidence, top-1/top-2 margin, or rejection-config distance '
                  'falls outside thresholds; your model does NOT need a '
                  'rejected class.',
            ],
          ),
          _section(
            theme,
            icon: Icons.shield_outlined,
            title: 'Rejection config (optional)',
            children: const [
              'If you also want to update the centroids used for the rejection '
                  'layer, replace `assets/model/rejection_config.json` in a '
                  'rebuild, or wire up the cloud meta JSON (see '
                  '`ModelManager.checkAndUpdateModel`).',
              'Loading just a `.tflite` is fine; the app keeps the previous '
                  'rejection config.',
            ],
          ),
          _section(
            theme,
            icon: Icons.science_outlined,
            title: 'Recommended training recipe',
            children: const [
              'Start from a Keras MobileNetV2 / EfficientNetB0 with '
                  '`include_top=False` and a small classification head.',
              'Add `keras.layers.Rescaling(1./255)` as the first layer so the '
                  'serving-time math matches the app.',
              'Augment with rotation, flip, brightness, and zoom to handle '
                  'phone-camera variance.',
              'Convert with: `tf.lite.TFLiteConverter.from_keras_model(model)` '
                  'and (optionally) `converter.optimizations = '
                  '[tf.lite.Optimize.DEFAULT]` for INT8 quantization.',
            ],
          ),
          _section(
            theme,
            icon: Icons.warning_amber_outlined,
            title: 'Common pitfalls',
            children: const [
              'Wrong input range (e.g. -1..1 or 0..255). Re-export with a '
                  'Rescaling(1/255) layer, or change the app preprocessing.',
              'Wrong channel order (BGR). Train with RGB.',
              'Different input resolution (e.g. 192×192). The app currently '
                  'feeds 224×224 only.',
              'Labels in a different order than the output channels. Re-export '
                  '`class_names.txt` in the order the model expects, or '
                  're-train with sorted class names.',
            ],
          ),
          _section(
            theme,
            icon: Icons.task_alt,
            title: 'What happens when you Load Model',
            children: const [
              '1. The picker returns bytes for your `.tflite`.',
              '2. The app validates by instantiating an interpreter.',
              '3. On success, a copy is saved into app storage and the path is '
                  'persisted in shared preferences.',
              '4. Subsequent launches automatically restore this model.',
              '5. On failure, the app falls back to the default bundled model '
                  'and shows the underlying error.',
            ],
          ),
          const SizedBox(height: 24),
          Container(
            decoration: BoxDecoration(
              color: theme.colorScheme.primaryContainer.withValues(alpha: 0.4),
              borderRadius: BorderRadius.circular(12),
            ),
            padding: const EdgeInsets.all(12),
            child: Text(
              'Tip: keep your training class order identical to '
              '`assets/model/class_names.txt`. If you rename classes '
              '(e.g. introduce a new stage), update the labels file in the '
              'project and re-bundle, or extend `ModelManager` to also '
              'load a user-supplied labels file alongside the model.',
              style: small,
            ),
          ),
        ],
      ),
    );
  }

  Widget _section(
    ThemeData theme, {
    required IconData icon,
    required String title,
    required List<String> children,
  }) {
    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(icon, size: 20, color: theme.colorScheme.primary),
                const SizedBox(width: 8),
                Text(title,
                    style: theme.textTheme.titleSmall
                        ?.copyWith(fontWeight: FontWeight.w700)),
              ],
            ),
            const SizedBox(height: 6),
            for (final line in children)
              Padding(
                padding: const EdgeInsets.only(left: 28, bottom: 4),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text('•   '),
                    Expanded(
                      child: Text(line, style: theme.textTheme.bodyMedium),
                    ),
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }
}

/// ===== HISTORY PAGE =====
class HistoryPage extends StatelessWidget {
  final List<HistoryItem> history;
  const HistoryPage({super.key, required this.history});

  @override
  Widget build(BuildContext context) {
    final colorScheme = Theme.of(context).colorScheme;
    return Scaffold(
      appBar: AppBar(title: const Text('History')),
      body: history.isEmpty
          ? const Center(child: Text('No history yet.'))
          : ListView.builder(
              padding: const EdgeInsets.all(12),
              itemCount: history.length,
              itemBuilder: (context, index) {
                final item = history[index];
                final badgeColor = classBadgeColor(
                  item.prediction.predictedClass,
                  colorScheme.primary,
                );
                return Card(
                  margin: const EdgeInsets.only(bottom: 10),
                  child: ListTile(
                    onTap: () => Navigator.of(context).push(
                      MaterialPageRoute(
                        builder: (_) => HistoryDetailPage(item: item),
                      ),
                    ),
                    leading: ClipRRect(
                      borderRadius: BorderRadius.circular(8),
                      child: Image.file(
                        File(item.imagePath),
                        width: 48,
                        height: 48,
                        fit: BoxFit.cover,
                        errorBuilder: (_, __, ___) =>
                            const Icon(Icons.broken_image, size: 32),
                      ),
                    ),
                    title: Text(
                      formatClassName(item.prediction.predictedClass),
                      style: TextStyle(
                        fontWeight: FontWeight.w600,
                        color: badgeColor,
                      ),
                    ),
                    subtitle: Text(
                      'Confidence: '
                      '${(item.prediction.confidence * 100).toStringAsFixed(1)}%\n'
                      '${item.timestamp}',
                    ),
                    isThreeLine: true,
                    trailing: const Icon(Icons.chevron_right),
                  ),
                );
              },
            ),
    );
  }
}

/// ===== HISTORY DETAIL PAGE =====
/// Shows the full "how did it get classified" breakdown behind one history
/// entry: every class's probability (not just the winner), and, when the
/// result was rejected, plain-language reasoning for why.
class HistoryDetailPage extends StatelessWidget {
  final HistoryItem item;
  const HistoryDetailPage({super.key, required this.item});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final p = item.prediction;
    final badgeColor =
        classBadgeColor(p.predictedClass, theme.colorScheme.primary);
    final readableClass = formatClassName(p.predictedClass);

    return Scaffold(
      appBar: AppBar(title: const Text('Scan Details')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Card(
            clipBehavior: Clip.antiAlias,
            child: Image.file(
              File(item.imagePath),
              width: double.infinity,
              height: 240,
              fit: BoxFit.cover,
              errorBuilder: (_, __, ___) => const SizedBox(
                height: 240,
                child: Center(child: Icon(Icons.broken_image, size: 48)),
              ),
            ),
          ),
          const SizedBox(height: 12),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(classIcon(p.predictedClass), color: badgeColor),
                      const SizedBox(width: 8),
                      Text(
                        readableClass,
                        style: theme.textTheme.titleLarge?.copyWith(
                          color: badgeColor,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const Spacer(),
                      Text(
                        '${(p.confidence * 100).toStringAsFixed(1)}%',
                        style: theme.textTheme.titleMedium?.copyWith(
                          color: badgeColor,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 4),
                  Text('${item.timestamp}', style: theme.textTheme.bodySmall),
                  const SizedBox(height: 12),
                  Text(p.explanation, style: theme.textTheme.bodyMedium),
                ],
              ),
            ),
          ),
          if (p.rejectionReason != null) ...[
            const SizedBox(height: 12),
            Card(
              color: Colors.red.withValues(alpha: 0.08),
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        const Icon(Icons.info_outline,
                            color: Colors.red, size: 18),
                        const SizedBox(width: 6),
                        Text(
                          'Why was this rejected?',
                          style: theme.textTheme.titleSmall?.copyWith(
                            color: Colors.red.shade700,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 6),
                    Text(p.rejectionReason!, style: theme.textTheme.bodyMedium),
                    if (p.rawPredictedClass != p.predictedClass) ...[
                      const SizedBox(height: 8),
                      Text(
                        'Closest match before rejection: '
                        '${formatClassName(p.rawPredictedClass)}',
                        style: theme.textTheme.bodySmall
                            ?.copyWith(fontStyle: FontStyle.italic),
                      ),
                    ],
                  ],
                ),
              ),
            ),
          ],
          if (p.classProbabilities.isNotEmpty) ...[
            const SizedBox(height: 12),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Classification Breakdown',
                      style: theme.textTheme.titleSmall
                          ?.copyWith(fontWeight: FontWeight.w700),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      'How confident the model was in each possible stage '
                      'for this image.',
                      style: theme.textTheme.bodySmall,
                    ),
                    const SizedBox(height: 12),
                    for (final entry in p.classProbabilities.entries)
                      Padding(
                        padding: const EdgeInsets.only(bottom: 10),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Row(
                              children: [
                                Text(
                                  formatClassName(entry.key),
                                  style: theme.textTheme.bodyMedium?.copyWith(
                                    fontWeight: entry.key == p.rawPredictedClass
                                        ? FontWeight.bold
                                        : FontWeight.normal,
                                  ),
                                ),
                                const Spacer(),
                                Text(
                                  '${(entry.value * 100).toStringAsFixed(1)}%',
                                  style: theme.textTheme.bodySmall,
                                ),
                              ],
                            ),
                            const SizedBox(height: 4),
                            ClipRRect(
                              borderRadius: BorderRadius.circular(4),
                              child: LinearProgressIndicator(
                                value: entry.value.clamp(0.0, 1.0),
                                minHeight: 8,
                                backgroundColor: theme.colorScheme.onSurface
                                    .withValues(alpha: 0.08),
                                valueColor: AlwaysStoppedAnimation<Color>(
                                  classBadgeColor(
                                      entry.key, theme.colorScheme.primary),
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                    const SizedBox(height: 4),
                    Text(
                      'Margin between top two classes: '
                      '${(p.margin * 100).toStringAsFixed(1)}%',
                      style: theme.textTheme.bodySmall,
                    ),
                  ],
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }
}

/// ===== BATCH PAGE =====
class BatchPage extends StatelessWidget {
  final List<BatchItem> batchResults;
  const BatchPage({super.key, required this.batchResults});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Batch Results')),
      body: batchResults.isEmpty
          ? const Center(child: Text('No batch results.'))
          : ListView.builder(
              itemCount: batchResults.length,
              itemBuilder: (context, index) {
                final item = batchResults[index];
                final readableClass = formatClassName(item.predictedClass);
                final pct = (item.confidence * 100).toStringAsFixed(1);
                final badgeColor = classBadgeColor(
                  item.predictedClass,
                  Theme.of(context).colorScheme.primary,
                );

                return Card(
                  margin:
                      const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                  child: ListTile(
                    leading: Image.file(
                      File(item.imagePath),
                      width: 56,
                      height: 56,
                      fit: BoxFit.cover,
                      errorBuilder: (_, __, ___) =>
                          const Icon(Icons.broken_image, size: 32),
                    ),
                    title: Text(readableClass,
                        style: TextStyle(
                            fontWeight: FontWeight.w600, color: badgeColor)),
                    subtitle: Text('Confidence: $pct%'),
                  ),
                );
              },
            ),
    );
  }
}

/// ===== INFO & MANUAL =====
class InfoPage extends StatelessWidget {
  const InfoPage({super.key});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(title: const Text('Information')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Text('Coconut Husk Maturity Stages',
              style: theme.textTheme.titleMedium),
          const SizedBox(height: 12),
          _gradeCard(
            theme,
            color: Colors.orange,
            icon: Icons.hourglass_top,
            title: 'Immature',
            lines: const [
              'Husk has not fully developed yet',
              'Fibers are underdeveloped and moisture content is high',
              'Best left to mature further before harvesting/processing',
            ],
          ),
          _gradeCard(
            theme,
            color: Colors.green,
            icon: Icons.star,
            title: 'Mature (Best)',
            lines: const [
              'Fully developed, high-quality fibers',
              'Ideal moisture content',
              'The ideal harvest stage for top processing quality',
            ],
          ),
          _gradeCard(
            theme,
            color: Colors.brown,
            icon: Icons.hourglass_bottom,
            title: 'Overmature',
            lines: const [
              'Husk has aged past its ideal stage',
              'Fibers may be dry or degraded',
              'Reduced processing quality',
            ],
          ),
          _gradeCard(
            theme,
            color: Colors.red,
            icon: Icons.block,
            title: 'Rejected',
            lines: const [
              'The image does not resemble a coconut husk, or quality is too low.',
              'Please recapture on a plain background with clear visibility.',
            ],
          ),
        ],
      ),
    );
  }

  Widget _gradeCard(
    ThemeData theme, {
    required Color color,
    required IconData icon,
    required String title,
    required List<String> lines,
  }) {
    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(icon, size: 20, color: color),
                const SizedBox(width: 8),
                Text(title,
                    style: theme.textTheme.titleSmall
                        ?.copyWith(fontWeight: FontWeight.w700, color: color)),
              ],
            ),
            const SizedBox(height: 6),
            for (final line in lines)
              Padding(
                padding: const EdgeInsets.only(left: 28, bottom: 4),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text('•   '),
                    Expanded(
                      child: Text(line, style: theme.textTheme.bodyMedium),
                    ),
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }
}

class ManualPage extends StatelessWidget {
  const ManualPage({super.key});

  static const List<String> _steps = [
    'Place the coconut husk on a plain, non-reflective background.',
    'Ensure good lighting; avoid strong shadows or glare.',
    'Tap "Live Camera" to open the in-app preview; the predicted stage '
        'updates roughly every 1.5 seconds. Aim the rear camera at the husk '
        'and tap "Capture" when the stage looks correct.',
    '"Gallery" lets you pick an existing photo. "Batch (Gallery)" analyzes '
        'many photos at once.',
    'Review the predicted maturity level and confidence. If the result is '
        '"Rejected", recapture a clearer husk image.',
    'Use the Information page to understand the meaning of each maturity '
        'level.',
    'You can review previous scans in the History section.',
    'To use a different model, open the menu and choose "Model Settings" '
        'to load a .tflite file from storage. See "Custom Model Guide" for '
        'the file format the app expects.',
  ];

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(title: const Text('User Manual')),
      body: ListView.builder(
        padding: const EdgeInsets.all(16),
        itemCount: _steps.length,
        itemBuilder: (context, index) => _stepCard(
          theme,
          number: index + 1,
          text: _steps[index],
        ),
      ),
    );
  }

  Widget _stepCard(ThemeData theme,
      {required int number, required String text}) {
    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            CircleAvatar(
              radius: 14,
              backgroundColor: theme.colorScheme.primary,
              child: Text(
                '$number',
                style: TextStyle(
                  color: theme.colorScheme.onPrimary,
                  fontWeight: FontWeight.w700,
                  fontSize: 13,
                ),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Text(text, style: theme.textTheme.bodyMedium),
            ),
          ],
        ),
      ),
    );
  }
}
