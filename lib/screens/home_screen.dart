import 'package:camera/camera.dart';
import 'package:face_recognition/screens/face_screen.dart';
import 'package:flutter/material.dart';
import 'package:gallery_saver/gallery_saver.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image_picker/image_picker.dart';

class Home extends StatefulWidget {
  const Home({Key? key, required this.title, required this.camera})
      : super(key: key);

  final String title;
  final CameraDescription camera;
  @override
  State<Home> createState() => _HomeState();
}

class _HomeState extends State<Home> {
  final ImagePicker _picker = ImagePicker();
  late XFile? pickedFile;
  late InputImage nInput;
  final FaceDetector faceDetector = FaceDetector(
    options: FaceDetectorOptions(performanceMode: FaceDetectorMode.accurate),
  );

  // allow to pick image from gallery
  Future pickCameraGallery() async {
    pickedFile = await _picker.pickImage(
        source: ImageSource.gallery, maxHeight: 640, maxWidth: 480);
    if (pickedFile == null) return;

    setState(() {
      pickedFile;
    });
  }

  // allow to pick image from camera feed
  Future pickCameraFeed() async {
    pickedFile = await _picker.pickImage(
        source: ImageSource.camera,
        preferredCameraDevice: CameraDevice.front,
        maxHeight: 640,
        maxWidth: 480);
    if (pickedFile == null) return;
    setState(() {
      pickedFile;
      GallerySaver.saveImage(pickedFile!.path).then((value) => {
            setState(() {
            
            })
          });
    });
     
  }

// build body
  @override
  Widget build(BuildContext context) {
    return Scaffold(
        appBar: AppBar(
          title: Text(widget.title),
        ),
        floatingActionButton: Column(
          mainAxisAlignment: MainAxisAlignment.end,
          children: [
            FloatingActionButton(
              onPressed: () async {
                try {
                  await pickCameraFeed();
                  if (!mounted) return;
                  await Navigator.of(context).push(
                    MaterialPageRoute(
                      builder: (context) => DisplayPictureScreen(
                        pickedFile: pickedFile,
                      ),
                    ),
                  );
                } catch (e) {
                  print(e);
                }
              },
              heroTag: null,
              child: const Icon(Icons.camera_alt_sharp),
            ),
            const SizedBox(
              height: 20,
            ),
            FloatingActionButton(
              onPressed: () async {
                try {
                  await pickCameraGallery();
                  if (!mounted) return;
                  await Navigator.of(context).push(
                    MaterialPageRoute(
                      builder: (context) => DisplayPictureScreen(
                        pickedFile: pickedFile,
                      ),
                    ),
                  );
                } catch (e) {
                  print(e);
                }
              },
              heroTag: null,
              child: const Icon(Icons.photo),
            )
          ],
        ),
        body: Container(
          width: MediaQuery.of(context).size.width,
          height: MediaQuery.of(context).size.height,
          padding: const EdgeInsets.only(top: 50),
          color: Colors.white,
          child: Column(
            mainAxisAlignment: MainAxisAlignment.start,
            children: [
              Image.asset(
                'assets/face.png',
                width: 300,
                height: 300,
              ),
              const SizedBox(
                height: 40,
              ),
              const Text(
                'Face AI',
                style: TextStyle(
                  fontSize: 30,
                  fontWeight: FontWeight.bold,
                ),
              )
            ],
          ),
        ));
  }
}
