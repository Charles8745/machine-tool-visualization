import cv2
from ultralytics import YOLO

# 載入你的模型（支援 yolov8/yolov11 格式）
model = YOLO("yolo_real_best.pt")

# 開啟攝影機
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("無法開啟攝影機")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 推論（OpenCV 預設是 BGR 格式，YOLOv8/11 會自動處理）
    results = model(frame)[0]

    # 畫出結果
    annotated_frame = results.plot()

    # 顯示畫面
    cv2.imshow("YOLOv11 Detection", annotated_frame)

    # 按 q 鍵離開
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 關閉攝影機與視窗
cap.release()
cv2.destroyAllWindows()
