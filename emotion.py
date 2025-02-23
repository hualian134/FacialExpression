import sys
import os

# Ensure stdout and stderr are properly initialized
if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w')
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w')

# Suppress TensorFlow and Keras logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QTextEdit, QMessageBox, QGridLayout
)
from PyQt6.QtGui import QImage, QPixmap, QFont, QIcon
from PyQt6.QtCore import QTimer, Qt
from mtcnn import MTCNN
from keras._tf_keras.keras.models import load_model


class EmotionDetectionApp(QWidget):
    def __init__(self):
        super().__init__()

        # Set the window icon
        self.setWindowIcon(QIcon("icons/emotion-recognition.ico"))
        self.setWindowTitle("Emotion Detection")
        self.setGeometry(100, 50, 1200, 700)

        # UI Elements
        self.title = QLabel("Facial Expression Recognition System\nGroup II")
        self.title.setStyleSheet("border:none; font-weight:Bold; color:gray; font-family:Arial; font-size:20px; margin:10px")
        self.title.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        self.video_label = QLabel(self)
        self.video_label.resize(800, 600)
        self.video_label.setStyleSheet("font-size: 20px; color: gray; border: 2px solid gray; font-weight:Bold")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        

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
        button_layout = QGridLayout()
        button_layout.addWidget(self.upload_image_button, 0, 0)
        button_layout.addWidget(self.upload_video_button, 0, 1)
        button_layout.addWidget(self.real_time_button, 0, 2)
        button_layout.addWidget(self.exit_button, 1, 1)

        self.member_layout = QHBoxLayout()
        teacher = QLabel("Supervised by:\nDr.Yu Yu Win")
        teacher.setStyleSheet("border:none")
        members = QLabel("Presented by:\nMKPT-6707 Ma Cho Mar Aye\nMKPT-7007 Mg Zaw Khant Win\nMKPT-7011 Ma Phoo Myat Thwe\nMKPT-7024 Ma Phyo Thazin")
        members.setStyleSheet("border: none")
        self.member_layout.addWidget(teacher)
        self.member_layout.addWidget(members)
        self.member_layout.setContentsMargins(100, 0, 50, 0)

        self.glayout = QGridLayout()
        self.glayout.addWidget(self.title, 0, 0)
        self.glayout.addLayout(self.member_layout, 0, 0)
        self.glayout.setSpacing(0)
        self.glayout.setContentsMargins(0, 0, 0, 0)
        self.video_label.setLayout(self.glayout)

        # Left Layout (Video Feed)
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.video_label)
        left_layout.addLayout(button_layout)

        # Right Layout (Emotion Display)
        right_layout = QVBoxLayout()
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
        self.model = load_model("models/emotion_detection_model.h5")
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    #delete inner layout
    def delete_layout(self):
        while self.member_layout.count():
            item = self.member_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater() #delete all widget 
        
        
        while self.glayout.count():
            item = self.glayout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater() #delete all widget 
        
    
    def upload_image(self):
        
        self.stop_video()
        self.stop_real_time_detection()
        if self.member_layout.count()>0:
            self.delete_layout()

        try:
            # Open file dialog to select an image
            file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg)")
            if file_path:
                # Load the image
                image = cv2.imread(file_path)
                if image is not None:
                    # Detect faces and emotions
                    processed_image = self.detect_faces_and_emotions(image)
                    self.display_image(processed_image)
                else:
                    raise ValueError("Failed to load the image. Please check the file path and format.")
        except Exception as e:
            # Display an error message to the user
            self.show_error_message(f"An error occurred: {str(e)}")

    def upload_video(self):

        self.stop_real_time_detection()
        if self.member_layout.count()>0:
            self.delete_layout()
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
        #Delete lable layout
        if self.member_layout.count()>0:
            self.delete_layout()

        # Start webcam feed
        self.cap = cv2.VideoCapture(0)
        self.timer.start(20)  # Update frame every 20ms

    def stop_real_time_detection(self):
        """Stop the real-time detection and release the camera."""
        if self.cap and self.cap.isOpened():
            self.timer.stop()
            self.cap.release()
            self.cap = None
            self.video_label.setText("Facial Expression Recognition System\nGroup II\n Supervised by Dr.Yu Yu Win\n")
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
        if not faces:
            emotion_text = "Emotion can't detect"
        else:
            emotion_text = "Emotion probabilities"
        for face in faces:
            x, y, width, height = face['box']
            face_region = frame[y:y + height, x:x + width]

            # Resize face region to match the input size of the emotion model
            resized_face = cv2.resize(face_region, (48, 48))
            gray_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)
            normalized_face = gray_face / 255.0
            input_data = np.expand_dims(np.expand_dims(normalized_face, -1), 0)

            # Predict emotion
            predictions = self.model.predict(input_data)
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
        try:
            # Convert the image to QImage
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(convert_to_Qt_format))
        except Exception as e:
            self.show_error_message(f"Failed to display the image: {str(e)}")

    def closeEvent(self, event):
        # Release resources when the window is closed
        if self.cap:
            self.cap.release()
        event.accept()

    def show_error_message(self, message):
        """Display an error message to the user."""
        error_box = QMessageBox()
        error_box.setIcon(QMessageBox.Icon.Critical)
        error_box.setWindowTitle("Error")
        error_box.setText(message)
        error_box.exec()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = EmotionDetectionApp()
    window.show()
    sys.exit(app.exec())