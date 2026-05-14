import socket
import cv2
import numpy as np
import time
import io
import threading
from PIL import Image
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS
from deep_translator import GoogleTranslator
from concurrent.futures import ThreadPoolExecutor

"""
SeeForMe AI Server
------------------
Handles real-time computer vision and image captioning for multiple concurrent 
UDP clients. Utilizes YOLOv8 for object detection and BLIP for scene description.
"""

# Global Configuration & Caching
label_cache = {}
COOLDOWN_SECONDS = 3

# AI Model Placeholders (Lazy Loading)
yolo_model = None
blip_processor = None
blip_model = None


def load_ai_models():
    # Initializes AI models only when the server begins processing frames
    # to conserve memory during idle startup.
    global yolo_model, blip_processor, blip_model
    if yolo_model is None:
        print("🧠 Loading YOLO & BLIP AI Models...")
        yolo_model = YOLO("yolov8l.pt")
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        print("✅ Models Loaded Successfully.")


class UserSession:
    # Maintains connection state and preferences for an individual client session.
    def __init__(self, address):
        self.address = address
        self.last_seen = time.time()
        self.last_spoken_time = 0
        self.language = 'en'
        self.is_slow_speech = False
        self.detail_level = 'Standard'
        self.lock = threading.Lock()


class MultiUserSessionManager:
    # Manages active user sessions, handles connection timeouts, and orchestrates
    # the thread pool for concurrent AI processing tasks.
    def __init__(self, timeout=10, max_workers=4):
        self.sessions = {}
        self.timeout = timeout
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def get_or_create_session(self, addr):
        # Retrieves an existing session or initializes a new one for a given address.
        if addr not in self.sessions:
            print(f"🟢 [NEW USER] Connected from {addr}")
            self.sessions[addr] = UserSession(addr)
        self.sessions[addr].last_seen = time.time()
        return self.sessions[addr]

    def clean_stale_sessions(self):
        # Removes sessions that have not transmitted data within the timeout window.
        current_time = time.time()
        stale = [addr for addr, sess in self.sessions.items() if current_time - sess.last_seen > self.timeout]
        for addr in stale:
            print(f"🟡 [USER TIMEOUT] Dropped {addr}")
            del self.sessions[addr]


def process_frame_for_user(payload, session, server_socket):
    # Core worker function executed in the thread pool.
    # Decodes incoming video frames, runs AI pipelines (BLIP & YOLO),
    # handles translation/TTS, and sends UDP packets back to the client.
    if not session.lock.acquire(blocking=False):
        return

    try:
        load_ai_models()

        np_arr = np.frombuffer(payload, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        current_time = time.time()

        # Scene Captioning & TTS (BLIP)
        if current_time - session.last_spoken_time > COOLDOWN_SECONDS:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            inputs = blip_processor(pil_image, return_tensors="pt")
            out = blip_model.generate(**inputs, max_new_tokens=30)
            english_desc = blip_processor.decode(out[0], skip_special_tokens=True)

            final_desc = english_desc
            if session.language != 'en':
                try:
                    final_desc = GoogleTranslator(source='en', target=session.language).translate(english_desc)
                except:
                    pass

            tts = gTTS(text=final_desc, lang=session.language, slow=session.is_slow_speech)
            mp3_fp = io.BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_bytes = mp3_fp.getvalue()

            if len(mp3_bytes) < 65000:
                audio_packet = b'\x01' + mp3_bytes
                server_socket.sendto(audio_packet, session.address)

            session.last_spoken_time = current_time

        # Object Detection (YOLOv8)
        results = yolo_model(frame, conf=0.35, verbose=False)
        detections = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxyn[0].tolist()
            english_label = results[0].names[int(box.cls[0])]

            cache_key = f"{session.language}_{english_label}"
            if cache_key in label_cache:
                final_label = label_cache[cache_key]
            else:
                try:
                    final_label = GoogleTranslator(source='en', target=session.language).translate(english_label)
                    label_cache[cache_key] = final_label
                except:
                    final_label = english_label

            new_x1, new_y1, new_x2, new_y2 = 1 - y2, x1, 1 - y1, x2
            detections.append(f"{final_label},{new_x1},{new_y1},{new_x2},{new_y2}")

        if detections:
            packet = b'\x03' + "|".join(detections).encode('utf-8')
            server_socket.sendto(packet, session.address)

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        session.lock.release()


if __name__ == "__main__":
    UDP_IP = "0.0.0.0"
    UDP_PORT = 5005
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.settimeout(1.0)

    session_manager = MultiUserSessionManager(timeout=10, max_workers=4)
    print(f"\n📡 SeeForMe AI Server listening on port {UDP_PORT}...")

    try:
        while True:
            try:
                session_manager.clean_stale_sessions()
                data, addr = sock.recvfrom(65535)
                session = session_manager.get_or_create_session(addr)

                packet_type = data[0]
                payload = data[1:]

                if packet_type == 2:  # COMMANDS
                    command = payload.decode('utf-8')
                    if command == "PING":
                        sock.sendto(b"PONG", addr)
                    elif command.startswith("LANG:"):
                        session.language = command.split(":")[1]
                elif packet_type == 1:  # VIDEO
                    session_manager.executor.submit(process_frame_for_user, payload, session, sock)

            except socket.timeout:
                continue
    except KeyboardInterrupt:
        print("\n🛑 Server stopped.")
        session_manager.executor.shutdown(wait=False)
    finally:
        sock.close()