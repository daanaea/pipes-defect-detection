## *PipeDetect: A Streamlit-Powered Web Tool for Automated Pipeline Defect Detection Using YOLOv11*

[![DOI](https://zenodo.org/badge/916820162.svg)](https://doi.org/10.5281/zenodo.15110765)


### *Abstract*

This paper presents PipeDetect, an open-access web-based platform for automated detection of pipeline defects using advanced deep learning techniques. Developed using Streamlit and YOLOv11, the system enables users to upload pipeline imagery and receive instant visual feedback on detected defects such as corrosion, deformation, and coating damage. The tool is designed for engineers, pipeline inspectors, and urban infrastructure managers, offering an accessible and intuitive interface without the need for specialized software or hardware. PipeDetect bridges the gap between cutting-edge computer vision models and real-world infrastructure monitoring by delivering AI-based defect analysis through the browser. The tool is deployed at [pipes-defect-detection.streamlit.app](https://pipes-defect-detection.streamlit.app) and the source code is available on GitHub [link].

---

### *1. Motivation and Significance*

Pipeline inspection remains a time-consuming and resource-intensive task, often relying on manual visual surveys that are prone to human error and inconsistency. With aging infrastructure and increasing environmental stressors (seismic activity, erosion, urban expansion), the need for scalable, fast, and accurate defect detection methods is critical.  
While deep learning-based models such as YOLO have shown strong performance in computer vision tasks, their integration into field-ready, user-friendly applications remains limited. PipeDetect addresses this gap by packaging a trained YOLOv11 model into an easy-to-use web interface, enabling non-technical users to conduct defect analysis without setup or installation.

---

### *2. Software Description*

#### 2.1 Software Architecture

The platform is composed of:
- *Frontend:* Built with Streamlit, providing an interactive UI for uploading images and displaying results.
- *Backend:* Utilizes YOLOv11 (Ultralytics implementation) for real-time object detection.
- *Data Pipeline:* Uploaded images are processed in memory, passed to the YOLOv11 model, and annotated with bounding boxes on detected defects.

#### 2.2 Functionality

Key features of PipeDetect:
- Upload any pipeline image (.jpg, .png)
- Detect defects in real-time using pre-trained model
- Display bounding boxes and classification labels for:
  - Corrosion
  - Bending
  - Deformation
- Export annotated images (future update)
- Simple web-based interface — no installation required

---

### *3. Illustrative Example*

A typical user workflow:
1. Visit [https://pipes-defect-detection.streamlit.app](https://pipes-defect-detection.streamlit.app)
2. Upload a photo of a gas pipeline
3. Within seconds, the system highlights areas of damage with labeled bounding boxes
4. Users visually inspect results or download screenshots

> In a pilot application in Almaty, Kazakhstan, the system was tested on field imagery collected from above-ground pipeline networks, achieving defect detection accuracy above 92%.

---

### *4. Impact*

- *Industrial Utility:* Assists technical personnel in gas distribution networks with rapid diagnostics.
- *Education & Research:* Provides a working demo for students and researchers studying applied machine vision.
- *Scalability:* Can be adapted for other infrastructures (oil, water pipelines, HVAC ducts).
- *Cross-Platform Access:* Works from any device with a browser; ideal for field inspectors with tablets.

---

### *5. Limitations and Future Improvements*

- Current model supports only image uploads (no video or live stream).
- Limited to predefined classes; retraining is needed for new defect types.
- Real-time video stream support is planned.
- Integration with drone-based imagery and edge computing is under development.

---

### *6. Conclusions*

PipeDetect exemplifies the practical use of AI and machine vision in industrial diagnostics. Its web-based nature makes it accessible to non-technical users, and its open-source model encourages community-driven improvements. Future iterations aim to expand defect classes, integrate predictive maintenance models, and offer API support for IoT systems.

---

### *Code Metadata Table*

| *Code metadata description* | *Details* |
|-------------------------------|-------------|
| Current code version | v1.0 |
| Codebase repository | [GitHub](https://github.com/daanaea/pipes-defect-detection/edit/main/README.md) |
| Legal Code License | MIT |
| Code versioning system used | Git |
| Software code languages, tools, and services used | Python, Streamlit, YOLOv11, OpenCV |
| Compilation requirements, operating environments & dependencies | Python 3.10, streamlit, torch, ultralytics |
| Support email for questions | eginovaa@gmail.com |

---
