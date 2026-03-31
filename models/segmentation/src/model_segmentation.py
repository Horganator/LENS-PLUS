import ultralytics
from ultralytics import YOLO
from IPython.display import Image 

ultralytics.checks()

model = YOLO("yolov8n-seg.pt")

print("Pre-trained classes", model.names.values())

results = model("")

results[0].save("{}")

Image(filename='/content/output.jpg')


for r in results:
    print(r.masks)