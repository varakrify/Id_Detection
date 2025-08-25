from ultralytics import YOLO

# Load pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")  # small, fast model

# Train on your dataset
model.train(
    data="./data.yaml",  # path to dataset.yaml
    epochs=50,                    # number of training epochs
    imgsz=640,                     # image size
    name="id_card_detector"        # folder to save trained model
)
