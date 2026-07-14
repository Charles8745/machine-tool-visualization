from ultralytics import YOLO
import cv2
import numpy as np
import socket
import struct
from wifi_tcp_helper import switch_wifi_and_send, receive_image, reture_original_wifi
from rl import rl_main
from machine_learning_model import pixel_to_real_position
from robot_ur import initialize_robot, move_joints, move_to, get_position
import math
import time

def detect_target(target_class, max_attempts=5):
    """Load model, receive image multiple times, and return first successful detection of target_class."""

    host = '192.168.50.46'
    port = 12345
    model_path = 'yolo_real_best.pt'

    def receive_image(sock):
        data_len = sock.recv(4)
        if not data_len:
            return None
        img_size = struct.unpack(">I", data_len)[0]

        data = b''
        while len(data) < img_size:
            packet = sock.recv(img_size - len(data))
            if not packet:
                return None
            data += packet

        img_array = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img

    model = YOLO(model_path)
    class_name_to_idx = {v: k for k, v in model.names.items()}

    if target_class not in class_name_to_idx:
        print(f"錯誤的類別名稱，請用以下其中一個：{list(class_name_to_idx.keys())}")
        return None, None, None

    target_class_idx = class_name_to_idx[target_class]

    for attempt in range(max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect((host, port))
                s.sendall(target_class.encode())
                img = receive_image(s)

            if img is None:
                print(f"[{attempt + 1}/{max_attempts}] 接收影像失敗")
                continue

            results = model(img)
            result = results[0]

            bboxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)

            conf_thresh = 0.8
            mask = (scores > conf_thresh) & (classes == target_class_idx)
            filtered_bboxes = bboxes[mask]

            if len(filtered_bboxes) == 0:
                print(f"[{attempt + 1}/{max_attempts}] 沒有偵測到 '{target_class}'")
                continue

            # 只取第一個框的中心點
            x1, y1, x2, y2 = filtered_bboxes[0].astype(int)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            crop = img[y1:y2, x1:x2].copy()
            crop_resized = cv2.resize(crop, (40, 40))

            # 標示結果
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(img, f"{target_class}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 存檔
            cv2.imwrite("detection_result.jpg", img)
            cv2.imwrite(f"{target_class}_crop_40x40.jpg", crop_resized)

            print(f"[{attempt + 1}/{max_attempts}] 偵測成功！")
            return cx, cy, crop_resized

        except Exception as e:
            print(f"[{attempt + 1}/{max_attempts}] 錯誤：{e}")
            continue

    print(f"未能在 {max_attempts} 次內成功偵測 '{target_class}'")
    return None, None, None



def main():
    target_class = "hammer"
    cx, cy, img = detect_target(target_class)

    if img is not None:
        print(f"偵測到 '{target_class}'，中心點座標：({cx}, {cy})")
        # 不用顯示，圖片已存在檔案中

        real_x_y_position = pixel_to_real_position(cx, cy)
        print(f"{target_class} 的實際位置為: {real_x_y_position}")

        rl_wrist3_radians_angle = rl_main(image=img)
    else:
        print(f"未能偵測到 '{target_class}' 或接收影像失敗")


if __name__ == "__main__":
    main()
