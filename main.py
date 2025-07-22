import cv2
from simple_facerec import SimpleFacerec



# Initialize face recognition system
sfr = SimpleFacerec()
sfr.load_encoding_images("dataset/", use_cache=True)

# Load camera
cap = cv2.VideoCapture(0)

print("[INFO] Starting camera... Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces
    face_locations, face_names = sfr.detect_known_faces(frame)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Draw rectangle and name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
