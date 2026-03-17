# Real-Time Drowsiness Detection System

##  Overview

This project is a **real-time driver drowsiness detection system** built using computer vision and facial landmark analysis. It monitors eye movement, blink rate, and head position to identify signs of fatigue and alert the user instantly.

The system is designed to help reduce road accidents by detecting early signs of driver drowsiness.

---

##  How It Works

* Captures live video using webcam
* Detects face using MediaPipe Face Mesh
* Extracts facial landmarks (eyes, mouth, head)
* Calculates:

  * **Eye Aspect Ratio (EAR)** → detects eye closure
  * **Blink Rate** → detects fatigue patterns
  * **Head Tilt** → detects nodding/sleeping posture
* Triggers alert if thresholds are crossed
* Logs events into a CSV file

---

##  Key Features

*  Eye closure detection using EAR
*  Blink rate monitoring with time window
*  Head tilt detection
*  Real-time logging (`drowsiness_log.csv`)
*  Fast and lightweight (runs on CPU)
*  Live webcam processing

---

##  Tech Stack

* **Python**
* **OpenCV** → Video processing
* **MediaPipe** → Facial landmark detection
* **NumPy** → Numerical computations
* **SciPy** → Distance calculation

---

## Project Structure

```
Drowsy Detection/
│
├── Live Drowsy Detection(1).py   # Main detection script
├── drowsiness_log.csv            # Event logs
├── venv/                         # Virtual environment (ignored)
└── .git/                         # Git files
```

---

## Installation & Setup

### 1. Clone the repository

```
git clone https://github.com/your-username/Drowsy-Detection.git
cd Drowsy-Detection
```

### 2. Install dependencies

```
pip install opencv-python mediapipe numpy scipy
```

### 3. Run the project

```
python "Live Drowsy Detection(1).py"
```

---

## Detection Parameters

| Parameter            | Description           | Value |
| -------------------- | --------------------- | ----- |
| EAR Threshold        | Eye closure detection | 0.25  |
| MAR Threshold        | Mouth open detection  | 0.75  |
| Blink Rate Threshold | Blinks per window     | 15    |
| Head Tilt Threshold  | Head angle            | 20    |

---

##  Output

* Logs stored in:

```
drowsiness_log.csv
```

* Example:

```
2026-02-18 10:42:47, Head Tilt Detected
```

---

##  Future Enhancements

*  Add real-time alarm sound
*  Mobile application integration
*  Deep learning model for higher accuracy
*  Web dashboard for monitoring
*  Cloud-based alert system

---

##  Limitations

* Works best in good lighting conditions
* Single-person detection only
* Depends on webcam quality

---

##  Author

**Mythreyan**

---

