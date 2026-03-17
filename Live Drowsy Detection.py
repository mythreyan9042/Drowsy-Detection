import cv2
import mediapipe as mp
import numpy as np
import subprocess
from scipy.spatial import distance
import threading
import time
import csv
from datetime import datetime
import queue
import os

# Constants
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.75
BLINK_RATE_THRESHOLD = 15
BLINK_WINDOW_SIZE = 10
HEAD_TILT_THRESHOLD = 20
DROWSINESS_TIME_THRESHOLD = 5

# Indices for landmarks
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [61, 39, 0, 269, 291, 405, 17, 181, 406, 313, 14, 87, 178, 402, 318, 324, 308]

class SpeechManager:
    def __init__(self):
        self.last_spoken_time = 0
        self.cooldown = 3
        self.is_speaking = False

    def speak_thread(self, message):
        self.is_speaking = True
        os.system(f'powershell -Command "Add-Type –AssemblyName System.Speech; '
                  f'(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{message}\');"')
        self.is_speaking = False

    def speak(self, message):
        current_time = time.time()

        if self.is_speaking:
            return

        if current_time - self.last_spoken_time < self.cooldown:
            return

        self.last_spoken_time = current_time

        threading.Thread(
            target=self.speak_thread,
            args=(message,),
            daemon=True
        ).start()
        
class DrowsinessDetector:
    def __init__(self):
        # Initialize speech manager
        self.speech_manager = SpeechManager()

        # Initialize MediaPipe Face Mesh
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

        # Initialize OpenCV Video Capture
        self.cap = cv2.VideoCapture(0)

        # Variables
        self.blink_count = 0
        self.blink_timer_start = time.time()   
        self.blink_detected = False            
        self.blink_times = []
        self.drowsiness_log = []
        self.eye_closed_start_time = None

        # Alert control variables
        self.last_drowsiness_alert = 0
        self.last_yawn_alert = 0
        self.last_head_tilt_alert = 0
        self.last_blink_alert = 0
        self.ALERT_REPEAT_INTERVAL = 2.5
        
        # Separate flags for each alert type
        self.drowsiness_active = False
        self.yawn_active = False
        self.head_tilt_active = False
        self.low_blink_active = False
        
        # Current alert message to display
        self.current_alert = ""

    def calculate_ear(self, eye_landmarks):
        poi_A = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
        poi_B = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
        poi_C = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
        ear = (poi_A + poi_B) / (2 * poi_C)
        return ear

    def calculate_mar(self, mouth_landmarks):
        poi_A = distance.euclidean(mouth_landmarks[1], mouth_landmarks[7])
        poi_B = distance.euclidean(mouth_landmarks[2], mouth_landmarks[6])
        poi_C = distance.euclidean(mouth_landmarks[3], mouth_landmarks[5])
        poi_D = distance.euclidean(mouth_landmarks[0], mouth_landmarks[4])
        mar = (poi_A + poi_B + poi_C) / (3 * poi_D)
        return mar

    def calculate_head_pose(self, face_landmarks, frame_shape):
        left_ear = face_landmarks.landmark[234]
        right_ear = face_landmarks.landmark[454]

        left_ear_px = (int(left_ear.x * frame_shape[1]), int(left_ear.y * frame_shape[0]))
        right_ear_px = (int(right_ear.x * frame_shape[1]), int(right_ear.y * frame_shape[0]))

        dx = right_ear_px[0] - left_ear_px[0]
        dy = right_ear_px[1] - left_ear_px[1]
        angle = np.degrees(np.arctan2(dy, dx))
        return angle

    def log_drowsiness(self, event):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.drowsiness_log.append((timestamp, event))
        with open("drowsiness_log.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, event])

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        # Reset current alert
        self.current_alert = ""

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_eye_points = self.extract_landmarks(face_landmarks, LEFT_EYE, frame.shape)
                right_eye_points = self.extract_landmarks(face_landmarks, RIGHT_EYE, frame.shape)
                mouth_points = self.extract_landmarks(face_landmarks, MOUTH, frame.shape)

                left_ear = self.calculate_ear(left_eye_points)
                right_ear = self.calculate_ear(right_eye_points)
                avg_ear = (left_ear + right_ear) / 2
                mar = self.calculate_mar(mouth_points)
                head_angle = self.calculate_head_pose(face_landmarks, frame.shape)

                if avg_ear < EAR_THRESHOLD:
                    if not hasattr(self, "blink_detected"):
                        self.blink_detected = False

                    if not self.blink_detected:
                        self.blink_detected = True
                else:
                    if hasattr(self, "blink_detected") and self.blink_detected:
                        self.blink_count += 1
                    self.blink_detected = False

                blink_rate = self.calculate_blink_rate()

                # DETECT ALL CONDITIONS INDEPENDENTLY
                self.detect_drowsiness(avg_ear)
                self.detect_yawn(mar)
                self.detect_head_tilt(head_angle)
                self.detect_low_blink()

                self.display_metrics(frame, avg_ear, mar, blink_rate, head_angle)

        # Display the alert
        self.display_alert(frame)
        
        return frame

    def extract_landmarks(self, face_landmarks, indices, frame_shape):
        points = []
        for idx in indices:
            x = int(face_landmarks.landmark[idx].x * frame_shape[1])
            y = int(face_landmarks.landmark[idx].y * frame_shape[0])
            points.append((x, y))
        return points

    def calculate_blink_rate(self):
        if len(self.blink_times) >= 2:
            return (len(self.blink_times) - 1) / (self.blink_times[-1] - self.blink_times[0]) * 60
        return 0

    def detect_drowsiness(self, avg_ear):
        current_time = time.time()

        if avg_ear < EAR_THRESHOLD:
            if self.eye_closed_start_time is None:
                self.eye_closed_start_time = current_time
            else:
                time_closed = current_time - self.eye_closed_start_time

                if time_closed >= DROWSINESS_TIME_THRESHOLD:
                    if not self.drowsiness_active:  # 🔥 Only trigger once
                        self.drowsiness_active = True
                        print("Drowsiness detected")
                        self.speech_manager.speak("Alert! Wake up!")
                        self.log_drowsiness("Drowsiness Detected")
        else:
            self.eye_closed_start_time = None
            self.drowsiness_active = False
            
    def detect_yawn(self, mar):
        if mar > MAR_THRESHOLD:
            if not self.yawn_active:
                self.yawn_active = True
                print("Yawning detected")
                self.speech_manager.speak("Alert! You are yawning.")
                self.log_drowsiness("Yawning Detected")
        else:
            self.yawn_active = False

    def detect_head_tilt(self, head_angle):
        if abs(head_angle) > HEAD_TILT_THRESHOLD:
            if not self.head_tilt_active:
                self.head_tilt_active = True
                print("Head tilt detected")
                self.speech_manager.speak("Alert! Keep your head straight.")
                self.log_drowsiness("Head Tilt Detected")
        else:
            self.head_tilt_active = False

    def detect_low_blink(self):
        current_time = time.time()

        if current_time - self.blink_timer_start >= 15:

            if self.blink_count < 3:
                print("Low blink detected")
                self.speech_manager.speak("Blink more often")
                self.low_blink_active = True
            else:
                self.low_blink_active = False

            self.blink_count = 0
            self.blink_timer_start = current_time

    def display_alert(self, frame):
        """Display the most important alert at bottom center."""
        height, width = frame.shape[:2]
        
        # Determine which alert to show
        if self.drowsiness_active:
            alert_text = "DROWSINESS DETECTED"
        elif self.head_tilt_active:
            alert_text = "HEAD TILT DETECTED"
        elif self.yawn_active:
            alert_text = "YAWN DETECTED"
        elif self.low_blink_active:
            alert_text = "LOW BLINK RATE"
        else:
            return
        
        # Display at bottom center
        text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (width - text_size[0]) // 2
        text_y = height - 50
        
        # Draw black background
        cv2.rectangle(frame, (text_x - 10, text_y - text_size[1] - 10), 
                     (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)
        # Draw alert text
        cv2.putText(frame, alert_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    def display_metrics(self, frame, avg_ear, mar, blink_rate, head_angle):
        # Display metrics in top left corner
        cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"MAR: {mar:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Blink: {blink_rate:.1f} bpm", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Head: {head_angle:.1f} deg", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def run(self):
        try:
            print("Starting Drowsiness Detection System...")
            print("Voice alerts should now work for all conditions")
            
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame = self.process_frame(frame)
                cv2.imshow("Drowsiness Detector", frame)

                if cv2.getWindowProperty("Drowsiness Detector", cv2.WND_PROP_VISIBLE) < 1:
                    break

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print(f"An error occurred: {e}")

        finally:
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = DrowsinessDetector()
    detector.run()