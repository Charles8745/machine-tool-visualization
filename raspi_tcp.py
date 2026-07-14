import socket
import cv2
import struct
import threading

HOST = '0.0.0.0'
PORT = 12345

# 有效指令列表
valid_commands = {"object1", "object2", "wrench", "hammer"}

# 全域變數：儲存目前攝影機畫面
latest_frame = None
frame_lock = threading.Lock()

def camera_loop():
    global latest_frame
    cap = cv2.VideoCapture(1)  # 改成你要的攝影機 index
    if not cap.isOpened():
        print("[SERVER] Camera not accessible.")
        return

    while True:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                latest_frame = frame.copy()

            cv2.imshow("Server Live Camera", frame)
            if cv2.waitKey(1) == 27:  # 按 ESC 離開
                break

    cap.release()
    cv2.destroyAllWindows()

def send_image(conn, img):
    result, img_encoded = cv2.imencode('.jpg', img)
    data = img_encoded.tobytes()
    conn.sendall(struct.pack(">I", len(data)))
    conn.sendall(data)

def tcp_server():
    global latest_frame

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"[SERVER] Listening on {HOST}:{PORT}")

        while True:
            conn, addr = s.accept()
            print(f"[SERVER] Connected by {addr}")
            with conn:
                try:
                    while True:
                        data = conn.recv(1024).decode().strip()
                        if not data:
                            print("[SERVER] Client disconnected.")
                            break

                        print(f"[SERVER] Received command: {data}")
                        if data in valid_commands:
                            with frame_lock:
                                if latest_frame is not None:
                                    send_image(conn, latest_frame)
                                    print(f"[SERVER] Image sent for command: {data}")
                                else:
                                    print("[SERVER] No frame available.")
                        else:
                            print(f"[SERVER] Unknown command: {data}")
                except Exception as e:
                    print(f"[SERVER] Error: {e}")
                    continue

def main():
    # 開啟攝影機顯示執行緒
    threading.Thread(target=camera_loop, daemon=True).start()

    # 主線程跑 TCP 伺服器
    tcp_server()

if __name__ == "__main__":
    main()
