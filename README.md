# SeeForMe: AI Driven Assistive Technology

**SeeForMe** is a real time AI assistant designed specifically for individuals with visual disabilities. It acts as a smart "digital eye," capturing live video streams from a smartphone and converting that visual data into spoken audio descriptions. By combining computer vision and natural language processing, the system analyzes the environment, identifies objects, and tells the user exactly what is happening around them in real time.

## 🚀 Key Features

* **Real Time Object Detection:** Utilizes **YOLOv8l** for high speed and accurate spatial labeling.
* **Scene Captioning:** Uses **BLIP** (Bootstrapping Language-Image Pre-training) to generate deep, contextual descriptions of the entire environment.
* **Multi User Scalability:** A robust Python backend manages concurrent sessions, isolating user data such as language and speech rate.
* **Low Latency Communication:** Implements a custom **UDP protocol** with a "Magic Byte" system to prioritize the most current frames and commands over packet reliability.
* **Accessibility First:** Designed according to **ISO 9241** standards, featuring high contrast themes (OLED Comfort, Low Glare) and haptic feedback.
* **Multilingual Support:** Integrated support for English, Turkish, German, and more via **gTTS** and **Deep Translator**.

## 🏗️ System Architecture

The project follows a **Client-Server architecture** optimized for mobile performance.

### Backend (Python)
* **Lazy Model Initialization:** To conserve resources, AI models are only loaded into memory upon the first client request.
* **Threaded Processing:** Utilizes a `ThreadPoolExecutor` to handle concurrent AI pipelines and UDP socket communication.
* **Confidence Tuning:** YOLOv8l is tuned to a **0.35 confidence threshold** to minimize false positives in critical navigation tasks.

### Mobile Client (Flutter)
* **Reactive UI:** Uses `ValueNotifiers` to update bounding boxes and status without interrupting the camera stream.
* **Hysteresis Flash Control:** Implements a software based algorithm to prevent camera flash flickering in low light environments by managing luminance thresholds.

## 🛠️ Used Technologies

* **Flutter & Dart:** For a responsive, cross platform mobile interface.
* **Python:** The core backend language for AI hosting and socket management.
* **Ultralytics YOLOv8l:** Advanced real time object detection.
* **Hugging Face Transformers (BLIP):** For image to text synthesis.
* **Google Text to Speech (gTTS):** For clear, multilingual audio feedback.

## 📊 Test Results

The system underwent rigorous validation according to **ISO/IEC/IEEE 29119** standards:
* **100% Pass Rate:** Successfully passed all 10 core functional test cases, including end to end workflow and network failure handling.
* **Automated Suites:** Verified by 17 automated mobile unit tests and 7 automated server unit tests.
* **Performance:** End to end latency remains consistently under the safety critical 4 second threshold.

## 📂 Installation & Setup

### Backend
1. Clone the repository and navigate to the server folder.
2. Install dependencies: `pip install -r requirements.txt`.
3. Start the server: `python Server.py`.

### Mobile
1. Install the Flutter SDK.
2. Connect an Android device and fetch dependencies: `flutter pub get`.
3. Run the application: `flutter run --release`.
4. Configure the Server IP in the application settings.

## 👥 The Team
* **İkra Yiğit Karaman**
* **Arda Erol**
* **Çağrı Demircan**

**Mentor:** Prof. Dr. Gökçe Nur Yılmaz

---
*Developed as a Final Project for the Departments of Software Engineering and Computer Engineering at TED University.*
