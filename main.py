import cv2
import pickle
import os
import time
import numpy as np
from deepface import DeepFace
from datetime import datetime
from firebase_config import db

def load_encodings():
    if not os.path.exists("encodings.pkl"):
        print("ERROR: encodings.pkl not found. Run register.py first.")
        return []
    with open("encodings.pkl", "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} registered student(s).")
    return data

def cosine_distance(a, b):
    a, b = np.array(a), np.array(b)
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def mark_attendance(name, marked_set):
    if name in marked_set:
        return
    marked_set.add(name)
    now = datetime.now()
    db.collection("attendance").add({
        "name": name,
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "timestamp": now.isoformat()
    })
    print(f"MARKED: {name} at {now.strftime('%H:%M:%S')}")

def run_attendance():
    data = load_encodings()
    if not data:
        return

    print("Opening camera...")

    # Try different backends and indexes
    cam = None
    for index in [0, 1, 2]:
        for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
            test = cv2.VideoCapture(index, backend)
            time.sleep(1)
            if test.isOpened():
                ret, frame = test.read()
                if ret and frame is not None and frame.sum() > 0:
                    print(f"Camera working: index={index}, backend={backend}")
                    cam = test
                    break
                test.release()
        if cam:
            break

    if cam is None:
        print("ERROR: Cannot get valid frame from any camera.")
        return

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Keep reading until we get a bright frame
    print("Warming up camera...")
    for i in range(30):
        ret, frame = cam.read()
        if ret and frame is not None and frame.sum() > 0:
            print(f"Camera ready after {i+1} frames!")
            break
        time.sleep(0.1)

    marked_today = set()
    frame_count  = 0
    last_name    = None
    last_box     = None

    print("Attendance system running. Press Q to quit.")

    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            print("Frame read failed, retrying...")
            time.sleep(0.1)
            continue

        # Always show the live frame immediately
        display = frame.copy()

        frame_count += 1

        # Run face recognition every 60 frames
        if frame_count % 60 == 0:
            try:
                temp = "temp_frame.jpg"
                cv2.imwrite(temp, frame)

                result = DeepFace.represent(
                    img_path=temp,
                    model_name="Facenet",
                    enforce_detection=True,
                    detector_backend="opencv"  # faster than default
                )

                if os.path.exists(temp):
                    os.remove(temp)

                embedding = result[0]["embedding"]
                area      = result[0]["facial_area"]

                # Find best match
                best_name = "Unknown"
                best_dist = float("inf")
                for person in data:
                    dist = cosine_distance(embedding, person["encoding"])
                    if dist < best_dist:
                        best_dist = dist
                        best_name = person["name"]

                if best_dist > 0.4:
                    best_name = "Unknown"

                last_name = best_name
                last_box  = area

                if best_name != "Unknown":
                    mark_attendance(best_name, marked_today)

            except Exception as e:
                if os.path.exists("temp_frame.jpg"):
                    os.remove("temp_frame.jpg")

        # Draw box on display frame
        if last_box and last_name:
            x = last_box["x"]
            y = last_box["y"]
            w = last_box["w"]
            h = last_box["h"]
            color = (0, 255, 0) if last_name != "Unknown" else (0, 0, 255)

            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(display, (x, y + h), (x + w, y + h + 35), color, cv2.FILLED)
            cv2.putText(display, last_name, (x + 6, y + h + 25),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        # Status bar
        status = f"Present: {len(marked_today)} | {datetime.now().strftime('%H:%M:%S')} | Q to quit"
        cv2.rectangle(display, (0, 0), (640, 35), (50, 50, 50), cv2.FILLED)
        cv2.putText(display, status, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Show live feed — this must always run
        cv2.imshow("Attendance System", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"\nSession ended. {len(marked_today)} student(s) marked present:")
    for n in sorted(marked_today):
        print(f"  - {n}")

if __name__ == "__main__":
    run_attendance()