import cv2
import numpy as np
from mtcnn import MTCNN
from keras._tf_keras.keras.models import load_model

# Load the pre-trained emotion detection model
emotion_model = load_model('emotion_detection_model.h5')  # Replace with your model path

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize MTCNN for face detection
detector = MTCNN()

def detect_emotion(image):
    """
    Detects faces in an image and predicts emotions for each face.
    Displays the emotion label and accuracy percentage for the predicted emotion.
    Also displays all possible emotion labels with their percentages in the top-right corner.
    """
    # Detect faces using MTCNN
    faces = detector.detect_faces(image)

    for face in faces:
        x, y, width, height = face['box']
        # Ensure the bounding box is within the image dimensions
        x, y = max(0, x), max(0, y)
        width, height = min(width, image.shape[1] - x), min(height, image.shape[0] - y)

        # Extract the face region
        face_img = image[y:y+height, x:x+width]

        # Preprocess the face image for emotion detection
        face_img = cv2.resize(face_img, (48, 48))  # Resize to 48x48
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        face_img = np.expand_dims(face_img, axis=-1)  # Add channel dimension
        face_img = np.expand_dims(face_img, axis=0)  # Add batch dimension
        face_img = face_img / 255.0  # Normalize

        # Predict emotion
        emotion_prediction = emotion_model.predict(face_img)
        emotion_label = emotion_labels[np.argmax(emotion_prediction)]
        accuracy = np.max(emotion_prediction) * 100  # Confidence score in percentage

        # Draw bounding box, emotion label, and accuracy percentage
        cv2.rectangle(image, (x, y), (x+width, y+height), (0, 255, 0), 2)
        label_text = f"{emotion_label} ({accuracy:.2f}%)"
        cv2.putText(image, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return image

def process_image(image_path):
    """
    Processes a single image for emotion detection.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}.")
        return

    # Detect emotions
    output_image = detect_emotion(image)

    # Display the result
    cv2.imshow('Emotion Detection - Image', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(video_path):
    """
    Processes a video for emotion detection.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video from {video_path}.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect emotions
        output_frame = detect_emotion(frame)

        # Display the result
        cv2.imshow('Emotion Detection - Video', output_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_real_time():
    """
    Processes real-time webcam feed for emotion detection.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect emotions
        output_frame = detect_emotion(frame)

        # Display the result
        cv2.imshow('Real-Time Emotion Detection', output_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function to continuously accept inputs
def main():
    while True:
        print("\nSelect mode:")
        print("1. Image")
        print("2. Video")
        print("3. Real-Time Webcam")
        print("4. Exit")
        mode = input("Enter mode (1/2/3/4): ")

        if mode == '1':
            image_path = input("Enter image path: ")
            process_image(image_path)
        elif mode == '2':
            video_path = input("Enter video path: ")
            process_video(video_path)
        elif mode == '3':
            process_real_time()
        elif mode == '4':
            print("Exiting the program.")
            break
        else:
            print("Invalid mode selected. Please try again.")

if __name__ == "__main__":
    main()