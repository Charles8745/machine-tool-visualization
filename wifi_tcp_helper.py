import socket
import subprocess
import time
import cv2
import numpy as np

# Function to switch to a specific WiFi and send a message to the target IP/port
def switch_wifi_and_send(host, port, ssid, message, timeout=10):
    s = None  # Initialize socket as None to ensure it can be closed properly even if an error occurs
    try:
    

        # Create TCP connection and send data
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        s.sendall(message.encode('utf-8'))
        print(f"📡 Sent: {message} to {host}:{port}")

        return True, f"✅ Message sent to {host}:{port}"

    except subprocess.CalledProcessError:
        return False, "❌ Failed to execute WiFi switch command."
    except Exception as e:
        return False, f"❌ TCP transmission failed: {e}"

    finally:
        if s:
            try:
                s.close()
                print("🔌 Socket closed.")
            except Exception as e:
                print(f"⚠️ Failed to close socket: {e}")



# Function to receive an image and coordinates via TCP socket
def receive_image(host='0.0.0.0', port=9999):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    print(f"📡 Listening on {host}:{port} for image and coordinates...")

    conn, addr = server_socket.accept()
    print(f"✅ Connected: {addr}")

    try:
        # 1. Receive center coordinates (terminated by '\n')
        coord_data = b""
        while not coord_data.endswith(b"\n"):
            byte = conn.recv(1)
            if not byte:
                raise Exception("❌ Failed to receive center coordinates.")
            coord_data += byte

        coord_str = coord_data.decode('utf-8').strip()
        print(f"📨 Raw coordinates string: {coord_str}")

        if "," not in coord_str:
            raise Exception("⚠️ Invalid coordinate format.")
        cx, cy = map(int, coord_str.split(','))
        print(f"🎯 Center coordinates: ({cx}, {cy})")

        # 2. Receive image length (4 bytes)
        img_len_bytes = conn.recv(4)
        if len(img_len_bytes) != 4:
            raise Exception("❌ Incomplete image length received.")
        img_len = int.from_bytes(img_len_bytes, byteorder='big')
        print(f"📦 Image size: {img_len} bytes")

        # 3. Receive image data
        img_data = b''
        while len(img_data) < img_len:
            packet = conn.recv(1024)
            if not packet:
                break
            img_data += packet

        if not img_data:
            raise Exception("❌ Image data is empty.")

        # 4. Decode the image
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise Exception("⚠️ Failed to decode image.")

        # # If you want to save or display image, uncomment below:
        # cv2.imwrite("received_image.jpg", img)
        # cv2.imshow("Received Image", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    except Exception as e:
        print(f"⚠️ Error occurred: {e}")
        return None, None, None

    finally:
        conn.close()
        print("🔌 Connection closed")

    return cx, cy, img

def is_connected_to(ssid):
    try:
        result = subprocess.run(
            ["netsh", "wlan", "show", "interfaces"],
            capture_output=True,
            text=True,
            encoding="utf-8"  # 防止 cp950 解碼錯誤
        )
        output = result.stdout
        if output:
            return ssid in output
        else:
            return False
    except Exception as e:
        print(f"⚠️ 無法檢查 WiFi 連線狀態：{e}")
        return False

def reture_original_wifi(restore_ssid):
    s = None
    while True:
        if is_connected_to(restore_ssid):
            print(f"✅ 已成功連線到 WiFi：{restore_ssid}")
            break  # 成功後跳出迴圈

        try:
            print(f"🔄 嘗試連線到 WiFi：{restore_ssid}...")
            subprocess.run(["netsh", "wlan", "connect", f"name={restore_ssid}"], check=True)
        except Exception as e:
            print(f"⚠️ 連線失敗：{e}")
        
        time.sleep(2)  # 每 2 秒重試一次，避免太頻繁

    # 關閉 socket（如有）
    if s:
        try:
            s.close()
            print("🔌 Socket closed.")
        except Exception as e:
            print(f"⚠️ Failed to close socket: {e}")