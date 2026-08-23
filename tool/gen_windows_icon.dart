// One-off utility: builds windows/runner/resources/app_icon.ico as a proper
// multi-resolution ICO (16/24/32/48/64/128/256 px) from huskpng.png.
//
// Why this exists: a hand-exported .ico that embeds only a single large
// frame (e.g. 256x256) has no small frame for Windows to use for the title
// bar/taskbar/Alt+Tab icon. Windows then crops the top-left corner of the
// large frame instead of scaling it down, which is why the app icon looked
// "zoomed into a corner" everywhere except the in-app widgets (which draw
// huskpng.png themselves via Image.asset and scale it correctly).
//
// Run with: dart run tool/gen_windows_icon.dart
// Re-run this whenever huskpng.png (the source artwork) changes.
import 'dart:io';

import 'package:image/image.dart' as img;

void main() {
  const sourcePath = 'huskpng.png';
  const outputPath = 'windows/runner/resources/app_icon.ico';
  const sizes = [16, 24, 32, 48, 64, 128, 256];

  final sourceBytes = File(sourcePath).readAsBytesSync();
  final source = img.decodeImage(sourceBytes);
  if (source == null) {
    stderr.writeln('Could not decode $sourcePath');
    exit(1);
  }

  final frames = [
    for (final size in sizes)
      img.copyResize(
        source,
        width: size,
        height: size,
        interpolation: img.Interpolation.average,
      ),
  ];

  final icoBytes = img.IcoEncoder().encodeImages(frames);
  File(outputPath).writeAsBytesSync(icoBytes);

  stdout.writeln('Wrote $outputPath with frames: $sizes '
      '(${icoBytes.length} bytes)');
}
