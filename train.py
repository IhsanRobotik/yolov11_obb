if __name__ == "__main__":
    from ultralytics import YOLO

    model = YOLO("yolo11n-obb.pt", task="obb")  
    results = model.train(data="./data.yaml", epochs=100)
