import ultralytics
from ultralytics import YOLO

model = YOLO("models/best.pt")
model = model.cuda()


model.predict("input_videos/test-2.mp4", conf=0.4, show=True)
