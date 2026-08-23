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

/// Where the currently active model came from.
enum ModelSource {
  bundled,
  cloud,
  local,
  custom,
}

class ModelManager {
  // model_meta.json, hosted on Google Drive by the client. Points at
  // model_url/labels_url/rejection_url, which in turn point at the actual
  // .tflite/class_names.txt/rejection_config.json files (verified 2026-08-23
  // to resolve to the same model/labels/rejection config already bundled
  // as the app's default).
  static const String _metaUrl =
      'https://drive.google.com/uc?export=download&id=1egap7mATjXnIzTTA6Bar70iL-elnYtGD';

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

  static Future<bool> hasLocalModel() async {
    final file = await _localModelFile();
    return file.exists();
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

  static Future<ModelUpdateStatus> checkAndUpdateModel() async {
    try {
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
      AppLogger.d(
          'ModelManager: local version=$currentVersion, remote version=$newVersion');

      if (newVersion <= currentVersion) {
        AppLogger.i(
            'ModelManager: local model is up to date (v$currentVersion).');
        return ModelUpdateStatus.upToDate;
      }

      AppLogger.i(
          'ModelManager: downloading new model version $newVersion ...');

      final List<Future<http.Response>> futures = [
        http.get(Uri.parse(modelUrl)),
        http.get(Uri.parse(labelsUrl)),
      ];

      if (rejectionUrl != null && rejectionUrl.isNotEmpty) {
        futures.add(http.get(Uri.parse(rejectionUrl)));
      }

      final responses = await Future.wait(futures);

      final modelResp = responses[0];
      final labelsResp = responses[1];
      http.Response? rejectionResp;
      if (rejectionUrl != null && rejectionUrl.isNotEmpty) {
        rejectionResp = responses[2];
      }

      if (modelResp.statusCode != 200 || labelsResp.statusCode != 200) {
        AppLogger.w(
          'ModelManager: download failed: model=${modelResp.statusCode}, '
          'labels=${labelsResp.statusCode}',
        );
        return ModelUpdateStatus.failed;
      }

      if (rejectionResp != null && rejectionResp.statusCode != 200) {
        AppLogger.w(
          'ModelManager: rejection_config download failed: '
          '${rejectionResp.statusCode} (continuing without it)',
        );
      }

      final modelFile = await _localModelFile();
      await modelFile.writeAsBytes(modelResp.bodyBytes, flush: true);

      final labelsFile = await _localLabelsFile();
      await labelsFile.writeAsString(labelsResp.body, flush: true);

      if (rejectionResp != null && rejectionResp.statusCode == 200) {
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
