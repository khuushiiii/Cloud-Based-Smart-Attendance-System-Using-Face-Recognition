import cv2
import pickle
import os
from deepface import DeepFace

def register_face():
    name = input("Enter student name: ").strip()
    if not name:
        print("Name cannot be empty.")
        return

    image_path = input("Enter image filename (e.g. khushi.jpg): ").strip().strip('"')

    if not os.path.exists(image_path):
        print(f"ERROR: File not found: {image_path}")
        return

    print("Processing image... (first time may take 1-2 minutes to download model)")

    try:
        result = DeepFace.represent(
            img_path=image_path,
            model_name="Facenet",
            enforce_detection=True
        )
        embedding = result[0]["embedding"]

    except Exception as e:
        print(f"ERROR: {e}")
        print("Use a clear, front-facing photo with good lighting.")
        return

    if os.path.exists("encodings.pkl"):
        with open("encodings.pkl", "rb") as f:
            all_data = pickle.load(f)
    else:
        all_data = []

    if any(d["name"] == name for d in all_data):
        print(f"'{name}' already exists. Updating.")
        all_data = [d for d in all_data if d["name"] != name]

    all_data.append({"name": name, "encoding": embedding})

    with open("encodings.pkl", "wb") as f:
        pickle.dump(all_data, f)

    print(f"SUCCESS: '{name}' registered! Total students: {len(all_data)}")


if __name__ == "__main__":
    while True:
        register_face()
        again = input("\nRegister another student? (y/n): ").strip().lower()
        if again != 'y':
            print("Registration complete.")
            break