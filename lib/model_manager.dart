import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'app_logger.dart';

enum ModelUpdateStatus {
  upToDate,
  updated,
  failed,
}

/// Reports load/download progress so the UI can show something more useful
/// than a static "Loading model..." message. [fraction] is 0.0-1.0 when
/// progress is actually measurable (a download with a known content length),
/// or null for an indeterminate phase (checking for updates, saving to
/// disk, verifying, etc).
typedef ModelLoadProgress = void Function(String phase, {double? fraction});

/// Where the currently active model came from.
enum ModelSource {
  bundled,
  cloud,
  local,
  custom,
}

class ModelManager {
  // model_meta.json, hosted in the husktechrepo/cocohusk GitHub repo (a
  // dedicated storage repo, not app source). Points at model_url/labels_url/
  // rejection_url, which are GitHub Release assets in the same repo
  // (verified 2026-08-26 to resolve to the same model/labels/rejection
  // config already bundled as the app's default).
  static const String _metaUrl =
      'https://raw.githubusercontent.com/husktechrepo/cocohusk/main/model_meta.json';

  static const String _localModelName = 'coconut_husk_quality_model.tflite';
  static const String _localLabelsName = 'class_names.txt';
  static const String _localRejectionCfgName = 'rejection_config.json';
  static const String _modelVersionKey = 'model_version';

  /// User-selected custom model path (absolute). When set and the file exists,
  /// it overrides every other source.
  static const String _customModelPathKey = 'custom_model_path';

  static Future<Directory> _appDir() async {
    return getApplicationDocumentsDirectory();
  }

  static Future<File> _localModelFile() async {
    final dir = await _appDir();
    return File('${dir.path}/$_localModelName');
  }

  static Future<File> _localLabelsFile() async {
    final dir = await _appDir();
    return File('${dir.path}/$_localLabelsName');
  }

  static Future<File> _localRejectionCfgFile() async {
    final dir = await _appDir();
    return File('${dir.path}/$_localRejectionCfgName');
  }

  // True only when the local cache is actually complete enough to be used:
  // both the model and its labels file must be present, since
  // readActiveModelBytes's "local" branch reads both. rejection_config.json
  // isn't required here — readLocalRejectionConfig already falls back to
  // the bundled one gracefully if it's missing.
  static Future<bool> hasLocalModel() async {
    final modelExists = await (await _localModelFile()).exists();
    final labelsExist = await (await _localLabelsFile()).exists();
    return modelExists && labelsExist;
  }

  static Future<Uint8List> readLocalModelBytes() async {
    final file = await _localModelFile();
    return file.readAsBytes();
  }

  static Future<String> readLocalLabels() async {
    final file = await _localLabelsFile();
    return file.readAsString();
  }

  static Future<String?> readLocalRejectionConfig() async {
    final file = await _localRejectionCfgFile();
    if (await file.exists()) {
      return file.readAsString();
    }
    return null;
  }

  static Future<int> getCurrentModelVersion() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getInt(_modelVersionKey) ?? 0;
  }

  static Future<void> _setCurrentModelVersion(int version) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setInt(_modelVersionKey, version);
  }

  // ---- Custom (user-selected) model ----

  /// Returns the persisted custom-model path if it exists on disk, else null.
  /// If the path was recorded but the file is gone, clears the preference.
  static Future<String?> getCustomModelPath() async {
    final prefs = await SharedPreferences.getInstance();
    final path = prefs.getString(_customModelPathKey);
    if (path == null || path.isEmpty) return null;
    if (await File(path).exists()) return path;
    AppLogger.w(
      'Custom model path "$path" no longer exists. Clearing preference.',
    );
    await prefs.remove(_customModelPathKey);
    return null;
  }

  static Future<void> setCustomModelPath(String path) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_customModelPathKey, path);
    AppLogger.i('Persisted custom model path: $path');
  }

  static Future<void> clearCustomModelPath() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove(_customModelPathKey);
    AppLogger.i('Cleared custom model preference; reverting to default.');
  }

  /// Copies the picked file into the app's documents dir so the bytes survive
  /// even if the original is moved/deleted by the user. Returns the new path.
  static Future<String> importCustomModel(File source) async {
    final dir = await _appDir();
    final fileName = source.uri.pathSegments.isNotEmpty
        ? source.uri.pathSegments.last
        : 'custom_model.tflite';
    final dest = File('${dir.path}/custom_$fileName');
    await source.copy(dest.path);
    AppLogger.i('Imported custom model to: ${dest.path}');
    return dest.path;
  }

  /// Writes raw bytes (e.g. from a Storage Access Framework picker that did
  /// not give us a usable filesystem path) into the app's documents dir.
  /// Returns the new path.
  static Future<String> writeCustomModelBytes({
    required Uint8List bytes,
    required String fileName,
  }) async {
    final dir = await _appDir();
    final safeName =
        fileName.endsWith('.tflite') ? fileName : '$fileName.tflite';
    final dest = File('${dir.path}/custom_$safeName');
    await dest.writeAsBytes(bytes, flush: true);
    AppLogger.i('Wrote custom model bytes to: ${dest.path} '
        '(${bytes.lengthInBytes} bytes)');
    return dest.path;
  }

  /// Reads the active model bytes based on this priority:
  /// 1. custom (user-loaded)  2. local-cached/cloud-updated  3. bundled asset.
  /// Returned [ModelSource] tells the caller which one was used.
  static Future<({Uint8List bytes, ModelSource source, String? path})>
      readActiveModelBytes({
    required Future<Uint8List> Function() readBundledAsset,
    ModelUpdateStatus? updateStatus,
  }) async {
    final customPath = await getCustomModelPath();
    if (customPath != null) {
      try {
        final bytes = await File(customPath).readAsBytes();
        AppLogger.i('Loaded model from CUSTOM path: $customPath '
            '(${bytes.lengthInBytes} bytes)');
        return (bytes: bytes, source: ModelSource.custom, path: customPath);
      } catch (e, st) {
        AppLogger.e(
          'Failed to read custom model at $customPath; falling back.',
          e,
          st,
        );
        await clearCustomModelPath();
      }
    }

    if (await hasLocalModel()) {
      final bytes = await readLocalModelBytes();
      final source = updateStatus == ModelUpdateStatus.updated
          ? ModelSource.cloud
          : ModelSource.local;
      AppLogger.i('Loaded model from ${source.name.toUpperCase()} cache '
          '(${bytes.lengthInBytes} bytes)');
      return (
        bytes: bytes,
        source: source,
        path: (await _localModelFile()).path
      );
    }

    final bytes = await readBundledAsset();
    AppLogger.i('Loaded model from BUNDLED asset '
        '(${bytes.lengthInBytes} bytes)');
    return (bytes: bytes, source: ModelSource.bundled, path: null);
  }

  /// Downloads [url] via a streamed request instead of a single-shot
  /// `http.get`, reporting byte-level progress through [onProgress] as it
  /// goes (when the server sends a Content-Length header; otherwise progress
  /// stays indeterminate). Used for the model file specifically, since that's
  /// the one large enough (tens of MB) to make a real difference to show.
  static Future<({int statusCode, Uint8List bytes})> _downloadWithProgress(
    String url, {
    required String phase,
    ModelLoadProgress? onProgress,
  }) async {
    final client = http.Client();
    try {
      final streamedResponse =
          await client.send(http.Request('GET', Uri.parse(url)));
      final total = streamedResponse.contentLength;
      final chunks = <int>[];
      var received = 0;
      onProgress?.call(phase, fraction: total != null ? 0.0 : null);
      await for (final chunk in streamedResponse.stream) {
        chunks.addAll(chunk);
        received += chunk.length;
        if (total != null && total > 0) {
          onProgress?.call(phase, fraction: received / total);
        }
      }
      return (
        statusCode: streamedResponse.statusCode,
        bytes: Uint8List.fromList(chunks),
      );
    } finally {
      client.close();
    }
  }

  /// Sniffs whether [bytes] look like an HTML page rather than the binary
  /// (or JSON/text) file that was actually requested — the signature of a
  /// host serving a sign-in/interstitial/error page with an HTTP 200 status
  /// instead of the real file.
  static bool _looksLikeHtml(Uint8List bytes) {
    if (bytes.isEmpty) return false;
    final headLen = bytes.length < 200 ? bytes.length : 200;
    final head = String.fromCharCodes(bytes.sublist(0, headLen)).toLowerCase();
    return head.contains('<!doctype html') || head.contains('<html');
  }

  static Future<ModelUpdateStatus> checkAndUpdateModel({
    ModelLoadProgress? onProgress,
  }) async {
    try {
      onProgress?.call('Checking for model updates...');
      final resp = await http.get(Uri.parse(_metaUrl));
      if (resp.statusCode != 200) {
        AppLogger.w('ModelManager: meta download failed: ${resp.statusCode}');
        return ModelUpdateStatus.failed;
      }

      final meta = jsonDecode(resp.body) as Map<String, dynamic>;
      final int newVersion = meta['version'] as int;
      final String modelUrl = meta['model_url'] as String;
      final String labelsUrl = meta['labels_url'] as String;
      final String? rejectionUrl =
          meta.containsKey('rejection_url') && meta['rejection_url'] != null
              ? meta['rejection_url'] as String
              : null;

      final currentVersion = await getCurrentModelVersion();
      final fileStillPresent = await hasLocalModel();
      AppLogger.d(
          'ModelManager: local version=$currentVersion, remote version=$newVersion, '
          'local file present=$fileStillPresent');

      // The version number alone isn't proof the cached model is actually
      // usable — it's a separate SharedPreferences value that survives even
      // if the cached .tflite file itself gets deleted (manually, storage
      // cleared, etc). Trusting the version alone here silently fell back
      // to the bundled asset with no warning logged, since nothing "failed"
      // from this function's point of view. Re-download whenever the file
      // is missing, regardless of what the version marker claims.
      if (newVersion <= currentVersion && fileStillPresent) {
        AppLogger.i(
            'ModelManager: local model is up to date (v$currentVersion).');
        return ModelUpdateStatus.upToDate;
      }
      if (newVersion <= currentVersion && !fileStillPresent) {
        AppLogger.w(
            'ModelManager: version record (v$currentVersion) says up to '
            'date, but the cached model file is missing on disk — '
            're-downloading.');
      }

      AppLogger.i(
          'ModelManager: downloading new model version $newVersion ...');

      // All three requests are kicked off together (same as before); only
      // the model download is streamed for progress, since it's the only
      // one large enough for that to matter.
      final modelDownload = _downloadWithProgress(
        modelUrl,
        phase: 'Downloading new model...',
        onProgress: onProgress,
      );
      final labelsFuture = http.get(Uri.parse(labelsUrl));
      final rejectionFuture = (rejectionUrl != null && rejectionUrl.isNotEmpty)
          ? http.get(Uri.parse(rejectionUrl))
          : null;

      final modelResult = await modelDownload;
      final labelsResp = await labelsFuture;
      final rejectionResp =
          rejectionFuture != null ? await rejectionFuture : null;

      if (modelResult.statusCode != 200 || labelsResp.statusCode != 200) {
        AppLogger.w(
          'ModelManager: download failed: model=${modelResult.statusCode}, '
          'labels=${labelsResp.statusCode}',
        );
        return ModelUpdateStatus.failed;
      }

      // A 200 status doesn't guarantee real content: some hosts (Google
      // Drive being the classic example) can serve an HTML sign-in or
      // interstitial page with a 200 status instead of the actual file,
      // e.g. if a "public" link's access is ever revoked or restricted.
      // Without this check that HTML would get treated as model bytes and
      // fail deep inside the TFLite interpreter with an opaque "not a
      // valid Flatbuffer buffer" error instead of a clear message here.
      if (_looksLikeHtml(modelResult.bytes) ||
          _looksLikeHtml(labelsResp.bodyBytes)) {
        AppLogger.w(
          'ModelManager: download returned HTML instead of the expected '
          'file (model_url/labels_url may no longer be publicly '
          'accessible - check the hosting repo/release visibility).',
        );
        return ModelUpdateStatus.failed;
      }

      final rejectionUsable = rejectionResp != null &&
          rejectionResp.statusCode == 200 &&
          !_looksLikeHtml(rejectionResp.bodyBytes);
      if (rejectionResp != null && !rejectionUsable) {
        AppLogger.w(
          'ModelManager: rejection_config download failed or returned HTML '
          '(status=${rejectionResp.statusCode}) - continuing without it.',
        );
      }

      onProgress?.call('Saving model locally...');

      final modelFile = await _localModelFile();
      await modelFile.writeAsBytes(modelResult.bytes, flush: true);

      final labelsFile = await _localLabelsFile();
      await labelsFile.writeAsString(labelsResp.body, flush: true);

      if (rejectionUsable) {
        final rejectionFile = await _localRejectionCfgFile();
        await rejectionFile.writeAsString(rejectionResp.body, flush: true);
        AppLogger.i('ModelManager: wrote local rejection_config.json.');
      }

      await _setCurrentModelVersion(newVersion);

      AppLogger.i('ModelManager: updated model to version $newVersion');
      return ModelUpdateStatus.updated;
    } catch (e, st) {
      AppLogger.e('ModelManager: checkAndUpdateModel error', e, st);
      return ModelUpdateStatus.failed;
    }
  }
}
