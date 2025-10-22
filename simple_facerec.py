import face_recognition
import cv2
import os
import numpy as np

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

    def load_encoding_images(self, images_path, use_cache=True):
        self.known_face_encodings = []
        self.known_face_names = []

        cache_dir = os.path.join(images_path, "_encodings")
        os.makedirs(cache_dir, exist_ok=True)

        print("[INFO] Loading known faces...")
        for person_name in os.listdir(images_path):
            person_folder = os.path.join(images_path, person_name)
            if not os.path.isdir(person_folder) or person_name == "_encodings":
                continue

            cache_file = os.path.join(cache_dir, f"{person_name}.npy")

            if use_cache and os.path.exists(cache_file):
                print(f"[CACHE] Loading {person_name} from {cache_file}")
                encodings = np.load(cache_file)
                for enc in encodings:
                    self.known_face_encodings.append(enc)
                    self.known_face_names.append(person_name.replace("_", " ").title())
                continue

            print(f"[ENCODE] Processing {person_name}...")
            person_encodings = []
            for img_file in os.listdir(person_folder):
                if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                img_path = os.path.join(person_folder, img_file)
                img = cv2.imread(img_path)
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                encs = face_recognition.face_encodings(rgb_img)
                if encs:
                    person_encodings.append(encs[0])
                    self.known_face_encodings.append(encs[0])
                    self.known_face_names.append(person_name.replace("_", " ").title())
                    print(f"  ✓ {img_file}")
                else:
                    print(f"  No face in {img_file}")

            if person_encodings:
                np.save(cache_file, np.array(person_encodings))
                print(f"[SAVED] {len(person_encodings)} encodings → {cache_file}")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        face_locations_rescaled = []

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            name = "Unknown"

            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

            face_names.append(name)

            top, right, bottom, left = face_location
            face_locations_rescaled.append((top * 4, right * 4, bottom * 4, left * 4))

        return face_locations_rescaled, face_names



