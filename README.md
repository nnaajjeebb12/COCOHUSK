# HuskTech: Coconut Husk Maturity Classifier

HuskTech is a Flutter application that classifies the maturity stage of
coconut husks from a photo using an on-device TensorFlow Lite image
classification model. The app runs fully offline by default and can
optionally pull a newer model from the cloud, or load a user-supplied
`.tflite` model from local storage.

The bundled model is trained to output one of three maturity stages. A
fourth outcome, `rejected`, is never produced by the model itself; the app
synthesizes it when a prediction doesn't look trustworthy.

| Stage        | Meaning                                                    |
|--------------|-------------------------------------------------------------|
| `immature`   | Husk not fully developed: underdeveloped fibers, high moisture |
| `mature`     | Ideal harvest stage: fully developed fibers, ideal moisture |
| `overmature` | Aged past its ideal stage: fibers may be dry or degraded    |
| `rejected`   | The image doesn't confidently match any known stage         |

The `rejected` outcome is produced by a small **rejection layer** on top of
the raw TFLite outputs, applied in order:

1. **Confidence threshold** (70%): the top prediction must be at least this
   confident.
2. **Top-1/top-2 margin** (20%): the top two classes must be separated by at
   least this much, or the model is treated as unable to tell them apart.
3. **Euclidean distance to class centroids**: even a confident, well-separated
   prediction is rejected if its full probability vector doesn't fall close
   to any known class's typical output, using centroids and per-class
   distance stats stored in `assets/model/rejection_config.json`.

See `lib/inference_engine.dart` for the exact logic, and the in-app
**Custom Model Guide** (menu → *Custom Model Guide*) for the full input/output
contract a custom model must satisfy.

---

## Features

- **On-device inference** with a TFLite EfficientNetB0-based classifier
  (224x224 input, transfer-learned; see "Model training & assets" below for
  the training recipe).
- **Live camera preview with continuous prediction** on Android: opens the
  camera inside the app, runs inference roughly every 1.5 seconds, and
  overlays the live stage + confidence. Tap **Capture** to lock the current
  frame's prediction (see `lib/live_camera_page.dart`). Hidden on platforms
  without a native camera plugin implementation (Windows).
- **Gallery / Batch** image input: still-image flows for analyzing existing
  photos or many photos at once.
- **Rejection layer**: flags out-of-distribution images instead of forcing a
  best-guess stage (see above).
- **Dynamic model loading**, in priority order:
  1. **Custom**: a `.tflite` picked in-app, persisted across restarts.
  2. **Cloud/local cache**: fetched via a versioned meta JSON
     (`lib/model_manager.dart`) and cached locally once downloaded.
  3. **Bundled**: the asset shipped inside the APK / EXE.
- **Model verification**: after every load, the app confirms the interpreter's
  actual input/output tensor shapes match what it expects (224x224x3 input,
  output class count matching the loaded labels) and surfaces a clear
  warning in **Model Settings** instead of silently mispredicting.
- **Reload / retry model**: a refresh action (app bar, the result card on
  load failure, and Model Settings) re-runs the full load path, useful for
  recovering from a failed load or picking up a custom model file that
  changed on disk.
- **History with classification breakdown**: every prediction is saved with
  its image and timestamp; tapping an entry shows every class's probability
  (not just the winner) and, when rejected, plain-language reasoning for why.
- **In-app Custom Model Guide**: explains the input/output contract, training
  recipe, and common pitfalls before you load a custom model
  (menu → *Custom Model Guide*).
- **Structured logging** via the [`logger`](https://pub.dev/packages/logger)
  package (`lib/app_logger.dart`), viewable in-app (menu → *View Logs*).
- Light / dark theme toggle.
- Per-stage explanation, in-app information, and user manual pages, each laid
  out as individually boxed cards.

---

## Project layout

```
lib/
  main.dart              UI shell, page routes, history, dynamic model UI
  inference_engine.dart  Owns the TFLite interpreter + rejection layer; predicts from bytes/file
  live_camera_page.dart  Live preview with periodic inference + final-capture handoff
  model_manager.dart     Bundled / cloud / custom model resolution + persistence
  log_page.dart          In-app log viewer
  app_logger.dart        Centralized logger wrapper
assets/model/
  coconut_husk_quality_model.tflite   Bundled default model (3-class)
  class_names.txt                     Labels, one per line, in output order
  rejection_config.json               Per-class centroids + distance stats
tool/
  gen_windows_icon.dart  Regenerates a proper multi-resolution windows/runner/resources/app_icon.ico from huskpng.png
android/                 Android platform code
windows/                 Windows platform code
```

> Only `android/` and `windows/` are configured in this project; there is no
> `ios/`, `macos/`, `linux/`, or `web/` folder. `_platformSupportsLiveCamera`
> in `main.dart` also checks `Platform.isIOS`, but that path is untested
> since no iOS project exists here.

Outside `lib/`, the project root also holds the **model training tooling**
(not part of the running app; see "Model training & assets" below):
`train_model.py`, `train_model_internet.py`, `convert_to_tflite.py`, and the
`dataset/` folder they read from. `CCH/` is a separate, standalone desktop
Tkinter tool for preparing training data and running these scripts from a
GUI; it isn't part of the Flutter app.

---

## Dependencies

Runtime (`pubspec.yaml`):

- `flutter` (SDK)
- `image_picker: ^1.0.7`: still-image camera/gallery access
- `camera: ^0.11.0`: live preview for continuous in-app prediction
- `tflite_flutter: ^0.12.0`: TFLite interpreter bindings
- `http: ^1.1.0`: used by `ModelManager` for optional cloud updates
- `path_provider: ^2.1.2`: locates the app documents directory
- `shared_preferences: ^2.2.2`: persists model version + custom model path
- `file_picker: ^8.1.4`: lets the user pick a `.tflite` from device storage
- `logger: ^2.4.0`: structured logging

Dev: `flutter_test`, `flutter_lints: ^4.0.0`, `flutter_launcher_icons: ^0.14.4`
(generates the Android launcher icon from `huskpng.png`), `image: ^4.9.2`
(used by `tool/gen_windows_icon.dart` to build the Windows `.ico`).

---

## Setup

1. **Install Flutter** 3.41 or later: https://docs.flutter.dev/get-started/install
2. From the project root, fetch packages:

   ```sh
   flutter pub get
   ```

3. Verify your toolchain:

   ```sh
   flutter doctor
   ```

   For Android builds you additionally need:
   - Android Studio (or stand-alone `cmdline-tools`) with **Android SDK
     Platform 34** and **Build-Tools 34.x**
   - **JDK 17 or newer** (AGP 8.x requires it; JDK 11 will fail)
   - `ANDROID_HOME` set, or run `flutter config --android-sdk <path>`

---

## Running

### Windows (development)

```sh
flutter clean
flutter run -d windows
```

> The **Live Camera** button is automatically hidden on Windows because
> neither `image_picker` nor the `camera` plugin has a native Windows
> implementation. Use **Gallery** instead. (This is also what fixes the
> original `Bad state: This implementation of ImagePickerPlatform requires
> a "cameraDelegate" in order to use ImageSource.camera` error.)
>
> If `flutter run` fails with a symlink or `PathExistsException` error, or
> the linker reports `cannot open ...husktech.exe for writing`, a previous
> run's `husktech.exe` is likely still running and holding the file. Close
> it (Task Manager, or `taskkill /F /IM husktech.exe`) and, if needed,
> `flutter clean` before retrying.

### Android (development on a device / emulator)

```sh
flutter run -d <android-device-id>
```

Find the device ID with `flutter devices`.

---

## Building

### Android APK (release)

Pre-requisites: JDK 17+, Android SDK Platform 34, valid `local.properties`
(`flutter pub get` writes this automatically once `ANDROID_HOME` is set).

```sh
flutter clean
flutter pub get
flutter build apk --release
```

On success the APK is written to:

```
build/app/outputs/flutter-apk/app-release.apk
```

For a smaller, ABI-split APK set:

```sh
flutter build apk --release --split-per-abi
# outputs: app-armeabi-v7a-release.apk, app-arm64-v8a-release.apk, app-x86_64-release.apk
```

> **Note:** the bundled model is ~43 MB, so a release APK runs well over
> 100 MB. The release build is currently signed with the debug keystore
> (see the `TODO` in `android/app/build.gradle.kts`), which is fine for
> sideloading but must be replaced with a real release keystore before any
> Play Store submission.

### Sideloading the APK on an Android phone

1. On the phone, enable **Settings → Apps → Special access → Install unknown
   apps** for whichever tool will transfer the APK (Files, Chrome, etc.).
2. Copy `app-release.apk` to the phone (USB, Drive, email, etc.).
3. Tap the file in the phone's Files app and follow the prompts.

To install over USB from a developer machine:

```sh
adb install -r build/app/outputs/flutter-apk/app-release.apk
```

### Windows desktop (release)

```sh
flutter build windows --release
```

Output: `build/windows/x64/runner/Release/husktech.exe` plus its runtime DLLs.

### App icon

The launcher/window icon is generated from `huskpng.png`, not hand-edited
per platform:

```sh
dart run flutter_launcher_icons   # regenerates android/app/src/main/res/mipmap-*/ic_launcher.png
dart run tool/gen_windows_icon.dart   # regenerates windows/runner/resources/app_icon.ico (multi-resolution)
```

Re-run both any time `huskpng.png` changes. `tool/gen_windows_icon.dart`
exists because `flutter_launcher_icons`'s own Windows output, and a
hand-exported single-size `.ico`, both leave Windows without a small icon
frame to use for the title bar/taskbar; it crops the large frame's corner
instead of scaling it.

---

## Live camera flow

Tap **Live Camera** on the home screen (Android) to open `LiveCameraPage`.
The page:

1. Enumerates available cameras and prefers the rear-facing one.
2. Starts a preview at `ResolutionPreset.medium` with audio disabled.
3. Runs a periodic timer (default **1500 ms**, see `_predictIntervalMs`) that
   captures a still via `CameraController.takePicture()`, reads the bytes,
   and feeds them to the shared `InferenceEngine`. The bottom overlay shows
   the live stage + confidence + explanation.
4. Tapping **Capture** stops the timer, takes a final still, runs one more
   inference, and returns the result to the home page where it's added to
   history and shown in the result card.

The page handles `AppLifecycleState` (releases the camera on pause, restarts
on resume) and surfaces friendly error messages for `CameraAccessDenied`.

If you need higher frame rates, lower `_predictIntervalMs`; mid-range phones
handle ~700 ms; flagships handle ~300 ms cleanly. Going below the device's
practical inference time just queues backlog and wastes battery.

---

## How dynamic model loading works

1. On startup, `ModelManager.readActiveModelBytes` resolves the active model
   in this priority order:
   1. **Custom**: a `.tflite` path persisted in `SharedPreferences` (set when
      the user picks a file in the app).
   2. **Local cache**: a previously cloud-downloaded model stored in the
      app's documents directory.
   3. **Bundled**: the asset shipped inside the APK / EXE.
2. Before resolving the above, `ModelManager.checkAndUpdateModel` fetches a
   meta JSON from `_metaUrl` (currently a client-hosted Google Drive file):

   ```json
   {
     "version": 1,
     "model_url": "...",
     "labels_url": "...",
     "rejection_url": "..."
   }
   ```

   If `version` is newer than the locally cached version, it downloads the
   three files and caches them, marking the model source as `cloud`. If the
   fetch fails (offline, 404, etc.) it silently falls back to whatever's
   already local or bundled; nothing blocks the user on a failed check.
3. The user can also open **Menu → Model Settings → Load a new .tflite
   model** (or the **Load Model** button on the home screen) to pick a file
   directly. The file is copied into the app's documents directory,
   validated by instantiating a TFLite `Interpreter` **and** checking its
   tensor shapes against what the app expects, and only then committed as
   the active model. A snackbar confirms: *"New model loaded:
   `<filename>`"*.
4. **Revert to default** clears the persisted custom path and reloads via
   the normal priority chain. **Reload current model** (Model Settings, or
   the app bar refresh icon) re-runs the whole load path without changing
   which source is active, useful after a failed load or an on-disk file
   change.
5. If a custom model fails to load (corrupt file, version mismatch, etc.)
   the app logs the error, drops the preference, and falls back to the
   default.

All transitions are logged via `AppLogger` at `info` / `warn` / `error`
level, viewable in-app via **View Logs**.

---

## Model training & assets

The running app only needs three files, declared as assets in
`pubspec.yaml`: `assets/model/coconut_husk_quality_model.tflite`,
`class_names.txt`, and `rejection_config.json`. Everything below is tooling
that *produces* those files; none of it ships in the app.

- **`train_model_internet.py`**: the current training recipe. Transfer-learns
  an `EfficientNetB0` (ImageNet weights) with a small classification head,
  reading from `dataset/train` / `dataset/val`.
- **`train_model.py`**: an older, simpler standalone CNN recipe (no
  pretrained weights). Kept for reference; not what produced the current
  bundled model.
- **`convert_to_tflite.py`**: converts a saved `.h5` Keras model to
  `.tflite` without re-running training.

**Critical preprocessing contract**: the app feeds the model **raw 0-255
pixel values**, unnormalized (`InferenceEngine._preprocessImage`). Any
model you train or convert must do its own rescaling/normalization as part
of the graph itself (e.g. a Keras `Rescaling(1/255)` layer, or relying on
`EfficientNet`'s built-in normalization the way `preprocess_input` does for
that family). Feeding a model that expects already-normalized 0-1 input,
without baking that normalization into the graph, will produce an
input-invariant model that predicts the same class regardless of the image.

`class_names.txt` must list labels one per line, in the exact order of the
model's output channels; the app does not need (and the model does not need
to be trained with) an explicit `rejected` class, since that outcome is
synthesized by the app's own rejection layer, not predicted by the model.

The full contract, with common pitfalls, is also in-app: menu → *Custom
Model Guide*.

---

## Android permissions

The `AndroidManifest.xml` declares:

- `INTERNET`, `ACCESS_NETWORK_STATE`: optional cloud model updates
- `CAMERA` (+ `uses-feature android.hardware.camera` non-required): camera input
- `READ_EXTERNAL_STORAGE` (maxSdkVersion 32): legacy file access; on
  Android 13+ the system picker grants per-file URI access without this
  permission

---

## Troubleshooting

- **`Bad state: ... cameraDelegate ...`**: you tried to use
  `ImageSource.camera` on a platform with no native camera implementation
  (Windows / Linux / macOS / web). On those platforms the Camera button is now
  hidden; use Gallery. On Android the button is shown and works natively.
- **`Unable to locate Android SDK`** during `flutter build apk`: install the
  Android SDK (Android Studio is the easiest path), then either set
  `ANDROID_HOME` or `flutter config --android-sdk <path>`.
- **`A problem occurred configuring project ':app'`** with a Java toolchain
  error: AGP 8.x needs JDK 17+. Point Flutter at a 17+ JDK with
  `flutter config --jdk-dir <path-to-jdk-17>`.
- **App boots but shows "Failed to load model"**: verify
  `assets/model/coconut_husk_quality_model.tflite` is present and listed under
  `flutter.assets` in `pubspec.yaml`. Check logs for the underlying error.
  Use the **Retry loading model** button on the result card, or the app bar
  refresh icon, to retry without restarting the app.
- **Log shows `verification warning: model outputs N classes but M labels
  are loaded`**: `class_names.txt` doesn't have the same number of lines as
  the active model's output layer. Fix the labels file (or the model) so
  the counts match; see Model Settings for the currently loaded tensor
  shapes and warning text.
- **`flutter run -d windows` fails to build / link**: see the note under
  "Running → Windows (development)" above; almost always a previous
  `husktech.exe` still holding the file.
- **Predictions always come back the same class regardless of the photo**:
  almost certainly the preprocessing mismatch described in "Model training
  & assets" above; a model expecting normalized 0-1 input will saturate
  into one constant output when fed raw 0-255 values (or vice versa). Check
  what your training script's first layer actually expects.
