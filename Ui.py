import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QLabel, QPushButton,QGridLayout, QVBoxLayout, QWidget, QFileDialog
)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import QTimer
from mtcnn import MTCNN
from keras._tf_keras.keras.models import load_model

class EmotionDetectionApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Emotion Detection")
        self.setGeometry(100, 100, 800, 600)

        # UI Elements
        self.label = QLabel(self)
        self.label.resize(800, 500)
        self.label.setText("Emotion Detection Feed Will Appear Here")
        self.label.setStyleSheet("font-size: 20px; color: gray;")

        # Buttons
        self.upload_image_button = QPushButton("Upload Image", self)
        self.upload_image_button.clicked.connect(self.upload_image)

        self.upload_video_button = QPushButton("Upload Video", self)
        self.upload_video_button.clicked.connect(self.upload_video)

        self.real_time_button = QPushButton("Real-Time Detection", self)
        self.real_time_button.clicked.connect(self.start_real_time_detection)

        self.exit_button = QPushButton("Exit", self)
        self.exit_button.clicked.connect(self.close)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.upload_image_button)
        layout.addWidget(self.upload_video_button)
        layout.addWidget(self.real_time_button)
        layout.addWidget(self.exit_button)

        #grid layout
        grid = QGridLayout()
        grid.addWidget(self.label, 0, 0, 1, 4)
        grid.addLayout(layout,1,1)
        self.setLayout(grid)

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
        # Open file dialog to select a video
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi)")
        if file_path:
            # Release any previous video capture
            if self.cap:
                self.cap.release()

            # Load the video
            self.cap = cv2.VideoCapture(file_path)
            self.timer.start(20)  # Update frame every 20ms

    def start_real_time_detection(self):
        # Release any previous video capture
        if self.cap:
            self.cap.release()

        # Start webcam feed
        self.cap = cv2.VideoCapture(0)
        self.timer.start(20)  # Update frame every 20ms

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Detect faces and emotions
                processed_frame = self.detect_faces_and_emotions(frame)
                self.display_image(processed_frame)

    def detect_faces_and_emotions(self, frame):
        global sorted_emotions
        # Detect faces using MTCNN
        faces = self.face_detector.detect_faces(frame)

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
            emotion = self.emotion_labels[np.argmax(predictions)]
            accuracy = np.max(predictions)*100

            # Draw bounding box and emotion label
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
            label = f"{emotion} ({accuracy:.2f})"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (25, 25, 255), 2, cv2.LINE_AA)

             # Sort emotions by percentage (highest to lowest)
            sorted_emotions = sorted(zip(self.emotion_labels, predictions[0]), key=lambda x: x[1],reverse=True)
            
            for label, pred in sorted_emotions:
            # for i, (label, pred) in enumerate(zip(emotion_labels, emotion_prediction[0])): #show all prediction
                text = f"{label}: {pred * 100:.2f}%"
                #cv2.putText(image, text, (image.shape[1] - 250, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (13, 4, 4), 1)
                #y_offset -= 25

                print(text)
        return frame

    def display_image(self, image):
        # Convert the image to QImage
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(convert_to_Qt_format))

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