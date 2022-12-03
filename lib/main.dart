import 'package:camera/camera.dart';
import 'package:face_recognition/screens/home_screen.dart';
import 'package:flutter/material.dart';


List<CameraDescription> cameras = [];

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  cameras = await availableCameras();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        primarySwatch: Colors.red,
      ),
      home: Home(title: 'Security System', camera: cameras[1]),
      debugShowCheckedModeBanner: false,
    );
  }
}
