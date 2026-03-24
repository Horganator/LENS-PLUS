import cv2
import os

output_dir = "frames"
os.makedirs(output_dir, exist_ok=True)

video = cv2.VideoCapture("D:\\Zayaan\\D_git\\LENS-PLUS\\models\\segmentation\\media\\cat.mp4")

if not video.isOpened():
    print("ERROR: Video could not be opened.")
    exit()

count = 0

while True:
    isSuccess, image = video.read()
    if not isSuccess:
        break

    cv2.imwrite(os.path.join(output_dir, f"frame{count}.jpg"), image)
    print(f"Frame {count} read - saved as - frame{count}.jpg")
    count += 1

video.release()