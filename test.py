# test.py
import cv2
import numpy as np
import dlib

img = cv2.imread("khushi.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.ascontiguousarray(img, dtype=np.uint8)

print(f"Shape: {img.shape}")
print(f"Dtype: {img.dtype}")
print(f"C-contiguous: {img.flags['C_CONTIGUOUS']}")
print(f"Min: {img.min()}, Max: {img.max()}")

detector = dlib.get_frontal_face_detector()
dets = detector(img, 1)
print(f"Faces found: {len(dets)}")