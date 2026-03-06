# 🎯 Face Recognition App

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![dlib](https://img.shields.io/badge/dlib-008000?style=for-the-badge&logoColor=white)

A Python-based computer vision application that **detects and identifies known individuals** in images and video using facial recognition. Built with OpenCV and the `face_recognition` library (powered by dlib), the app matches detected faces against a set of known people in real time.

---

## ✨ Features

- 🧠 **Face Detection** — Locates faces in images and video frames using dlib's HOG-based detector
- 🪪 **Face Identification** — Matches detected faces against a dataset of known individuals
- 📸 **Image & Video Support** — Works on static images and live or pre-recorded video
- 🏷️ **Live Labeling** — Draws bounding boxes and name labels directly on recognized faces
- ⚡ **Fast Encoding** — Generates 128-dimensional face embeddings for accurate, efficient matching
- 📁 **Extendable Dataset** — Easily add new known faces by dropping images into the dataset folder

---

## 🚀 Demo

> The app scans each frame, encodes detected faces, and compares them against known encodings — labeling matches by name in real time.

![App Demo](https://your-demo-gif-or-screenshot-here.gif)


---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Language** | Python 3 |
| **Computer Vision** | OpenCV (`cv2`) |
| **Face Recognition** | `face_recognition` library (dlib) |
| **ML Model** | dlib's ResNet-based 128-D face embedding model |
| **Detection Method** | HOG + Linear SVM face detector |

---

## 🧠 How It Works

```
Input (image or video)
        │
        ▼
  Face Detection  ──────────────── OpenCV + dlib HOG detector
        │
        ▼
  Face Encoding   ──────────────── 128-D vector per face (dlib ResNet)
        │
        ▼
  Compare Encodings  ───────────── Euclidean distance vs. known faces
        │
        ▼
  Label & Display  ─────────────── Bounding box + name overlay via OpenCV
```

1. **Load known faces** from the dataset folder and compute their 128-D encodings
2. **Detect faces** in each input image or video frame
3. **Encode detected faces** into feature vectors
4. **Compare** against known encodings using Euclidean distance
5. **Label** matched faces with their name; unrecognized faces marked as `"Unknown"`

---

## 📂 Project Structure

```
face-recognition-app/
├── known_faces/             # Reference images — one photo per known person
│   ├── person_name.jpg
│   └── ...
├── encode_faces.py          # Pre-encodes all known faces into embeddings
├── recognize_faces.py       # Main script — runs detection + recognition
├── utils.py                 # Helper functions
├── requirements.txt
└── README.md
```

---

## ⚙️ Getting Started

### Prerequisites

- Python 3.7+
- `cmake` installed (required to build dlib)
- A C++ compiler (gcc on Linux/Mac, or Visual Studio Build Tools on Windows)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/carlotarzua/face-recognition-app.git
   cd face-recognition-app
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Key packages:
   ```
   face_recognition
   opencv-python
   numpy
   dlib
   ```

### Usage

**1. Add known faces**
Drop one clear, front-facing photo per person into the `known_faces/` folder. Name each file after the person (e.g., `carlota.jpg`).

**2. Encode known faces**
```bash
python encode_faces.py
```

**3. Run on an image**
```bash
python recognize_faces.py --image path/to/image.jpg
```

**4. Run on a video or webcam**
```bash
python recognize_faces.py --video path/to/video.mp4
# Live webcam:
python recognize_faces.py --video 0
```

---

## 🌱 Future Improvements

- [ ] Add a web interface with Flask or FastAPI
- [ ] Switch to CNN-based detector for better accuracy (GPU-accelerated)
- [ ] In-app face registration — no manual file management
- [ ] Export recognition logs with timestamps
- [ ] Containerize with Docker for easy deployment

---

## 👩‍💻 About Me

Built by **Carlota Arzúa** — a developer passionate about computer vision and applied machine learning.

- 💼 [LinkedIn](https://www.linkedin.com/in/carlota-a-53a75b206/)
- 🌐 [Portfolio]()
- 📧 carlotaarzua@email.com

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
