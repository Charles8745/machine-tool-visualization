import cv2
import requests
import numpy as np
from ultralytics import YOLO

# 載入 YOLO 模型
model = YOLO('yolo_real_best.pt')
CONFIDENCE_THRESHOLD = 0.75

# 類別名稱與顏色對應
CLASS_COLORS = {
    0: (0, 255, 0),    # hammer - Green
    1: (0, 0, 255),    # object1 - Red
    2: (255, 0, 0),    # object2 - Blue
    3: (0, 255, 255),  # wrench - Yellow
}

# Wrist Camera IP 地址，請根據你的相機 IP 修改
wrist_camera_ip = "192.168.0.102"

# 用來從 wrist camera 獲取影像的函數
def get_wrist_camera_frame():
    try:
        url = f"http://{wrist_camera_ip}:4242/current.jpg?type=color"
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return image
        else:
            print(f"❌ HTTP 錯誤，狀態碼: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ 無法從 Wrist Camera 取得影像: {e}")
        return None

# 主迴圈
while True:
    # 獲取 wrist camera 影像
    frame = get_wrist_camera_frame()
    if frame is None:
        continue

    # 進行 YOLO 偵測
    results = model(frame)

    for result in results:
        if not hasattr(result, 'boxes'):
            continue

        for box in result.boxes:
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            if conf < CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            color = CLASS_COLORS.get(cls, (255, 255, 255))

            # 繪製框與標籤
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f'{model.names[cls]} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.circle(frame, (cx, cy), 5, color, -1)

    # 顯示畫面
    cv2.imshow("Wrist Camera YOLO Detection", frame)

    # 按下 q 鍵離開
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 清理資源
cv2.destroyAllWindows()
