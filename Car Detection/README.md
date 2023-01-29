## Car Detection using YOLO
***
### **Objective**
Object detection is key to the next generation of autonomous cars. Machine learning and classic computer vision which uses object detection generally suffers from the moderate response interval. Recent and modern algorithms based on artificial neural networks, such as the You Only Look Once (YOLO) Algorithm, which can solve this problem without precision losses. This project focuses on the construction of a Deep Learning (DL) system for detecting car objects in images. YOLO (You only look once) can aid in determining whether an image contains a car or not, and if so, where it is located and how many cars are present. In this project, two YOLO variants, YOLOv5 and YOLOv7 are used, and fitted them to our previously collected and annotated dataset. Then, using the testing dataset, we test and evaluate them, finally, we select our champion model.

###**Dataset**

![](https://github.com/TokaMamdoh/Computer_Vision/blob/920d66f074a33ea5c545ae1c8b32aadf59f98ae0/Car%20Detection/images/car%201.PNG)
![](https://github.com/TokaMamdoh/Computer_Vision/blob/920d66f074a33ea5c545ae1c8b32aadf59f98ae0/Car%20Detection/images/car%202.PNG)

car dataset are gathered from google and mixed datasets with varying sources, Camera angles, resolutions and background that contain other vehicles and multiple object. This dataset contains 500 images, 350 training images and 100 validation images and about 50 test images.

### **Breif introduction of YOLO**
YOLO (You Only Look Once) is a method that provides real-time object detection using neural networks. It has been utilized in a variety of applications to find people, animals, parking meters, and traffic signals. YOLO performs object detection as a regression problem and outputs the class probabilities of the detected images. Convolutional neural networks (CNN) are used by the YOLO method to recognize objects in real time. The approach just needs one forward propagation through a neural network to detect objects, as the name would imply. This indicates that a single algorithm run is used to perform prediction throughout the full image. Multiple class probabilities and bounding boxes are simultaneously predicted using CNN. The YOLO algorithm become popular because of its speed, high accuracy, and learning capability. YOLO architectures consist of three main parts which are the backbone, neck, and head. The Backbone primarily pulls out the most important aspects of an image and transmits them through the Neck to the Head. The Neck compiles feature maps that the Backbone has retrieved and builds feature pyramids. The head's output layers have final detections, and they are the last layer. 

![](https://github.com/TokaMamdoh/Computer_Vision/blob/69efdd38b1e9f8250b4fa13e9e596aa8dd2b460c/Car%20Detection/images/Yolo%20architecture.PNG)

### ** YOLOv5**
Yolov5 has five versions which are extra-small (nano), small (s), medium (m), large (l), and extra-large (x) offering increasingly higher accuracy rates. The five models are identical in terms of the operations carried out, except for the number of layers and parameters. All the YOLOv5 models are composed of the same 3 components: CSP-Darknet53 as a backbone, SPP and PANet in the model neck, and the head used in YOLOv4. 

![](https://github.com/TokaMamdoh/Computer_Vision/blob/69efdd38b1e9f8250b4fa13e9e596aa8dd2b460c/Car%20Detection/images/YOLOv5%20architecture.PNG)

###**YOLOv7**
YOLOv7 introduces several architectural improvements that increase efficiency and accuracy. YOLOv7 backbones do not use ImageNet pre-trained backbones, like Scaled YOLOv4. Instead, the complete COCO dataset is used to train the models. As Scaled YOLOv4, an extension of YOLOv4 was produced by the same authors as YOLOv7, the similarity is to be expected. 

###**Result**

| Models  Metrics | YOLOv5 | YOLOv7|
| ---             | ---    | ---   |
| Precision       | 81.9%  | 78.7% |
| Recall          | 80.6%  | 65.6% |
| mAP             | 85.2%  | 72.6% |

- Train result for YOLOv5

![](https://github.com/TokaMamdoh/Computer_Vision/blob/69efdd38b1e9f8250b4fa13e9e596aa8dd2b460c/Car%20Detection/images/train%20result%20for%20YOLOv5.PNG)

- Train result for YOLOv7

![](https://github.com/TokaMamdoh/Computer_Vision/blob/69efdd38b1e9f8250b4fa13e9e596aa8dd2b460c/Car%20Detection/images/train%20result%20for%20YOLOv7.PNG)

- F1-Confidence curve for YOLOv5

![](https://github.com/TokaMamdoh/Computer_Vision/blob/69efdd38b1e9f8250b4fa13e9e596aa8dd2b460c/Car%20Detection/images/F1-Confidence%20curve%20YOLOv5.PNG)

- Precision-Recall curve for Yolov5

![](https://github.com/TokaMamdoh/Computer_Vision/blob/69efdd38b1e9f8250b4fa13e9e596aa8dd2b460c/Car%20Detection/images/Precision-Recall%20curve.PNG)

- F1-Confidence curve for YOLOv7

![](https://github.com/TokaMamdoh/Computer_Vision/blob/69efdd38b1e9f8250b4fa13e9e596aa8dd2b460c/Car%20Detection/images/F1-Confidence%20curve%20YOLOv7.PNG)

- Precision-Recall curve for Yolov7

![](https://github.com/TokaMamdoh/Computer_Vision/blob/69efdd38b1e9f8250b4fa13e9e596aa8dd2b460c/Car%20Detection/images/Precision-Recall%20curve%20YOLOv7.PNG)
