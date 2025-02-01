import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QTextEdit
)
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtCore import QTimer
from mtcnn import MTCNN
from keras._tf_keras.keras.models import load_model

class EmotionDetectionApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Emotion Detection with MTCNN")
        self.setGeometry(100, 100, 1200, 800)

        # UI Elements
        self.video_label = QLabel(self)
        self.video_label.resize(800, 600)
        self.video_label.setText("Emotion Detection Feed Will Appear Here")
        self.video_label.setStyleSheet("font-size: 20px; color: gray; border: 2px solid gray;")

        # Emotion Display Area
        self.emotion_display = QTextEdit(self)
        self.emotion_display.setReadOnly(True)
        self.emotion_display.setFont(QFont("Arial", 12))
        self.emotion_display.setStyleSheet("background-color: #f0f0f0; padding: 10px; border: 2px solid gray;")

        # Buttons
        self.upload_image_button = QPushButton("Upload Image", self)
        self.upload_image_button.setStyleSheet("font-size: 14px; padding: 10px;")
        self.upload_image_button.clicked.connect(self.upload_image)

        self.upload_video_button = QPushButton("Upload Video", self)
        self.upload_video_button.setStyleSheet("font-size: 14px; padding: 10px;")
        self.upload_video_button.clicked.connect(self.upload_video)

        self.real_time_button = QPushButton("Real-Time Detection", self)
        self.real_time_button.setStyleSheet("font-size: 14px; padding: 10px;")
        self.real_time_button.clicked.connect(self.start_real_time_detection)

        self.exit_button = QPushButton("Exit", self)
        self.exit_button.setStyleSheet("font-size: 14px; padding: 10px; background-color: #ff4444; color: white;")
        self.exit_button.clicked.connect(self.close)

        # Layouts
        main_layout = QHBoxLayout()

        # Left Layout (Video Feed)
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.video_label)
        left_layout.addWidget(self.upload_image_button)
        left_layout.addWidget(self.upload_video_button)
        left_layout.addWidget(self.real_time_button)
        left_layout.addWidget(self.exit_button)

        # Right Layout (Emotion Display)
        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Emotion Probabilities:"))
        right_layout.addWidget(self.emotion_display)

        # Add left and right layouts to main layout
        main_layout.addLayout(left_layout, 70)  # 70% width for video feed
        main_layout.addLayout(right_layout, 30)  # 30% width for emotion display

        self.setLayout(main_layout)

        # Video capture and timer
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Load MTCNN for face detection
        self.face_detector = MTCNN()

        # Load emotion detection model
        self.emotion_model = load_model("emotion_detection_model.h5")  # Replace with your model path
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def upload_image(self):
        
        self.stop_video()
        self.stop_real_time_detection()
        # Open file dialog to select an image
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_path:
            # Load the image
            image = cv2.imread(file_path)
            if image is not None:
                # Detect faces and emotions
                processed_image = self.detect_faces_and_emotions(image)
                self.display_image(processed_image)

    def upload_video(self):

        self.stop_real_time_detection()
        # Open file dialog to select a video
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi)")
        if file_path:
            # Release any previous video capture
            if self.cap:
                self.cap.release()

            # Load the video
            self.cap = cv2.VideoCapture(file_path)
            self.timer.start(20)  # Update frame every 20ms
    
    def stop_video(self):
        self.timer.stop()
            

    def start_real_time_detection(self):
        # Release any previous video capture
        self.stop_video()
        if self.cap:
            self.cap.release()

        # Start webcam feed
        self.cap = cv2.VideoCapture(0)
        self.timer.start(20)  # Update frame every 20ms

    def stop_real_time_detection(self):
        """Stop the real-time detection and release the camera."""
        if self.cap and self.cap.isOpened():
            self.timer.stop()
            self.cap.release()
            self.cap = None
            self.video_label.setText("Emotion Detection Feed Will Appear Here")
            self.video_label.setStyleSheet("font-size: 20px; color: gray; border: 2px solid gray;")


    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Detect faces and emotions
                processed_frame = self.detect_faces_and_emotions(frame)
                self.display_image(processed_frame)

    def detect_faces_and_emotions(self, frame):
        # Detect faces using MTCNN
        faces = self.face_detector.detect_faces(frame)

        emotion_text = ""
        for face in faces:
            x, y, width, height = face['box']
            face_region = frame[y:y + height, x:x + width]

            # Resize face region to match the input size of the emotion model
            resized_face = cv2.resize(face_region, (48, 48))
            gray_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)
            normalized_face = gray_face / 255.0
            input_data = np.expand_dims(np.expand_dims(normalized_face, -1), 0)

            # Predict emotion
            predictions = self.emotion_model.predict(input_data)
            # emotion_index = np.argmax(predictions)
            # emotion = self.emotion_labels[emotion_index]
            # confidence = predictions[emotion_index]
            emotion = self.emotion_labels[np.argmax(predictions)]
            accuracy = np.max(predictions)*100

            # Draw bounding box and emotion label
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(frame, f"{emotion} ({accuracy:.2f}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            sorted_emotions = sorted(zip(self.emotion_labels, predictions[0]), key=lambda x: x[1],reverse=True)
            # Prepare emotion probabilities text
            emotion_text += f"Face at ({x}, {y}):\n"
            for label, pred in sorted_emotions:
            #for label, pred in zip(self.emotion_labels, predictions):
                emotion_text += f"{label}: {pred*100:.2f}%\n"
            emotion_text += "\n"

        # Update emotion display
        self.emotion_display.setText(emotion_text)

        return frame

    def display_image(self, image):
        # Convert the image to QImage
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(convert_to_Qt_format))

    def closeEvent(self, event):
        # Release resources when the window is closed
        if self.cap:
            self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EmotionDetectionApp()
    window.show()
    sys.exit(app.exec())