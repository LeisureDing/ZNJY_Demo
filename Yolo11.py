import cv2
from ultralytics import YOLO
import time

def YOLOv11(model, image, classes, save_path=None):
    results = model(image, verbose=False)
    detected_safe = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            if conf > 0.8:
                classname = classes[cls] if cls < len(classes) else "Unknown"
                centerx = int((xyxy[2] + xyxy[0]) / 2)
                centery = int((xyxy[3] + xyxy[1]) / 2)
                detected_safe.append((classname, (centerx, centery), conf))
                
                cv2.rectangle(image, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                cv2.putText(image, f"{classname} {conf:.2f}", (int(xyxy[0]), int(xyxy[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    if save_path:
        cv2.imwrite(save_path, image)

if __name__ == '__main__':
    model = YOLO("/home/orangepi/SmartSave/Saving_demo/weight/yolov11/Balls.pt")
    with open('/home/orangepi/SmartSave/Saving_demo/weight/class1.txt', "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    loop_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        loop_count += 1
        start_time = time.time()  

        save_path = f"'/home/orangepi/SmartSave/Saving_demo/PredResult/detct_{loop_count}.jpg"  
        YOLOv11(model, frame, classes, save_path)

        end_time = time.time()  
        elapsed_time = end_time - start_time  

        print(f'第 {loop_count} 次循环运行时间: {elapsed_time:.3f} 秒')

    cap.release()
    cv2.destroyAllWindows()
