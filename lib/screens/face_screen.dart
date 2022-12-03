import 'dart:io';
import 'dart:math';
import 'package:face_recognition/utils/utils.dart';
import 'package:flutter_image_compress/flutter_image_compress.dart';
import 'package:gallery_saver/gallery_saver.dart';
import 'package:image/image.dart' as img;
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:path/path.dart';
import 'package:path_provider/path_provider.dart';
import 'package:tflite/tflite.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';

class DisplayPictureScreen extends StatefulWidget {
  final XFile? pickedFile;

  const DisplayPictureScreen({super.key, required this.pickedFile});

  @override
  State<DisplayPictureScreen> createState() => _DisplayPictureScreenState();
}

class _DisplayPictureScreenState extends State<DisplayPictureScreen> {
  File? _image;
  File? _faceImagePath;
  String? confidence = '';
  String? name = '';
  String? id = '';
  String? info = '';
  var details = {
    0: 'Secuestro',
    1: 'Robo',
    2: 'Asalto',
    3: 'Asalto',
    4: 'Boleta de captura',
    5: 'Alimentos',
    6: 'Narcotrafico',
    7: 'Asalto',
    8: 'Prohibicion salida del país',
    9: 'Asesinato'
  };
  var ids = {
    0: '100234959',
    1: '173473747',
    2: '173939329',
    3: '049990099',
    4: '100394959',
    5: '094483454',
    6: '100344549',
    7: '174513433',
    8: '202202822',
    9: '149838484'
  };
  final FaceDetector faceDetector = FaceDetector(
    options: FaceDetectorOptions(performanceMode: FaceDetectorMode.accurate),
  );

  @override
  void initState() {
    super.initState();
    loadModel();
    _processPickedFile(widget.pickedFile);
  }

  @override
  void dispose() {
    Tflite.close();
    super.dispose();
  }

  // Load model.ftlite
  Future loadModel() async {
    try {
      String res;
      res = (await Tflite.loadModel(
        model: "assets/model.tflite",
        labels: "assets/labels.txt",
        // useGpuDelegate: true,
      ))!;
    } on PlatformException {
      print('Failed to load model.');
    }
  }

  // process image file obtained from gallery or camera
  Future _processPickedFile(XFile? pickedFile) async {
    final path = pickedFile?.path;
    if (path == null) {
      return;
    }
    setState(() {
      _image = File(path);
    });
    final saveDir = await getApplicationDocumentsDirectory();
    final targetPath = join(saveDir.path, pickedFile?.name);
    await FlutterImageCompress.compressAndGetFile(_image!.path, targetPath);

    final inputImage = InputImage.fromFilePath(targetPath);

    detectFace(inputImage);
  }

  // run face detector, preprocess image and try inference
  Future detectFace(InputImage inputImage) async {
    final List<Face> faces = await faceDetector.processImage(inputImage);

    if (mounted) {
      setState(() {
        print(faces);
      });
    }

    var imageBytes = (await rootBundle.load(inputImage.filePath!)).buffer;
    img.Image? oriImage = img.decodeJpg(imageBytes.asUint8List());

    for (Face face in faces) {
      double x, y, w, h;
      x = (face.boundingBox.left);
      y = (face.boundingBox.top);
      w = (face.boundingBox.width);
      h = (face.boundingBox.height);
      img.Image croppedImage =
          img.copyCrop(oriImage!, x.round(), y.round(), w.round(), h.round());
      croppedImage = img.copyResize(croppedImage, height: 224, width: 224);
      Directory documentDirectory = await getApplicationDocumentsDirectory();

      File file = File(join(documentDirectory.path, inputImage.filePath));
      file.writeAsBytesSync(img.encodePng(croppedImage));
      GallerySaver.saveImage(file.path).then((value) => {
            setState(() {
              _faceImagePath = file;
            })
          });
      classifyImage(croppedImage);
    }
  }

// run classification inference
  Future classifyImage(img.Image image) async {
    var output = await Tflite.runModelOnBinary(
      binary: imageToByteListFloat32(image, 224, 0, 255),
      numResults: 10,
      threshold: 0.05,
    );
    setState(() {
      if (output![0]['label'] != null) {
        name = output[0]['label'];
        confidence =
            truncateToDecimalPlaces(output[0]['confidence'], 3).toString();
        id = ids[output[0]['index']];
        info = details[output[0]['index']];
      }
      print(output);
    });
  }

  double truncateToDecimalPlaces(num value, int fractionalDigits) =>
      (value * pow(10, fractionalDigits)).truncate() /
      pow(10, fractionalDigits);
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Information')),
      body: Container(
        width: MediaQuery.of(context).size.width,
        height: MediaQuery.of(context).size.height,
        padding: const EdgeInsets.only(top: 50),
        color: Colors.white,
        child: Column(
          children: [
            _faceImagePath != null
                ? Image.file(
                    _faceImagePath!,
                    width: 250,
                    height: 250,
                    fit: BoxFit.cover,
                  )
                : Container(
                    margin: const EdgeInsets.only(top: 80),
                    width: 70,
                    height: 70,
                    child: const CircularProgressIndicator(),
                  ),
            const SizedBox(
              height: 40,
            ),
            Row(
              children: [
                const SizedBox(
                  width: 20,
                ),
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text('Confidence:',
                        style: TextStyle(
                            fontWeight: FontWeight.bold, fontSize: 20)),
                    const SizedBox(
                      height: 5,
                    ),
                    Text(confidence!,
                        style: const TextStyle(
                            fontWeight: FontWeight.normal, fontSize: 20)),
                    const SizedBox(
                      height: 10,
                    ),
                    const Text('Full name:',
                        style: TextStyle(
                            fontWeight: FontWeight.bold, fontSize: 20)),
                    const SizedBox(
                      height: 5,
                    ),
                    Text(name!,
                        style: const TextStyle(
                            fontWeight: FontWeight.normal, fontSize: 20)),
                    const SizedBox(
                      height: 10,
                    ),
                    const Text('ID Number:',
                        style: TextStyle(
                            fontWeight: FontWeight.bold, fontSize: 20)),
                    const SizedBox(
                      height: 5,
                    ),
                    Text(id!,
                        style: const TextStyle(
                            fontWeight: FontWeight.normal, fontSize: 20)),
                    const SizedBox(
                      height: 10,
                    ),
                    const Text('Info:',
                        style: TextStyle(
                            fontWeight: FontWeight.bold, fontSize: 20)),
                    const SizedBox(
                      height: 5,
                    ),
                     Text(info!,
                        style: const TextStyle(
                            fontWeight: FontWeight.normal, fontSize: 20))
                  ],
                ),
              ],
            )
          ],
        ),
      ),
    );
  }
}
