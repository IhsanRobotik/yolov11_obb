if __name__ == "__main__":
    from ultralytics import YOLO

    # Load pretrained YOLOv11 OBB model
    model = YOLO("yolo11n-obb.pt", task="obb")  # explicitly set task to 'obb'

    # Train on your dataset (make sure data.yaml is properly formatted)
    results = model.train(data="./data.yaml", epochs=100)
