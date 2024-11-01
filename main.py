import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def image_detection(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found")
        return

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(rgb_image)
        if results.detections:
            print("Face detected.")
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)
            cv2.imshow('image Face Detection', image)
            cv2.waitKey(0)
        else:
            print("No face detected.")

    cv2.destroyAllWindows()

image_detection('arman.jpg')
