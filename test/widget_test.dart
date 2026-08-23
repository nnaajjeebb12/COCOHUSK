import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:husktech/main.dart';

void main() {
  testWidgets('App boots and shows the title', (WidgetTester tester) async {
    await tester.pumpWidget(const HuskTechApp());
    await tester.pump();

    expect(find.byType(MaterialApp), findsOneWidget);
  });
}
