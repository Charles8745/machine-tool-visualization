from wifi_tcp_helper import switch_wifi_and_send, receive_image, reture_original_wifi
from rl import rl_main
from machine_learning_model import pixel_to_real_position
from robot_ur import initialize_robot, move_joints, move_to, get_position
import math
import time
import requests

task_list = ["hammer"]
#wrench hammer
def run_missions(target_missions):
    mir_ip = "192.168.50.26"
    host = f"http://{mir_ip}/api/v2.0.0/"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Basic RGlzdHJpYnV0b3I6NjJmMmYwZjFlZmYxMGQzMTUyYzk1ZjZmMDU5NjU3NmU0ODJiYjhlNDQ4MDY0MzNmNGNmOTI5NzkyODM0YjAxNA=='
    }

    # 取得所有任務資料
    res = requests.get(host + "missions", headers=headers)
    missions = res.json()

    for target_name in target_missions:
        mission = next((m for m in missions if m['name'] == target_name), None)

        if not mission:
            print(f"⚠️ 找不到任務名稱: {target_name}")
            continue

        print(f"\n▶️ 加入任務: {mission['name']} ({mission['guid']})")
        response = requests.post(
            url=host + "mission_queue",
            headers=headers,
            json={"mission_id": mission['guid']}
        )
        print("  狀態碼:", response.status_code)
        print("  回傳:", response.text)

        if response.status_code == 201:
            mission_queue = response.json()
            mission_id = mission_queue.get('id')

            print("⌛ 等待機器人完成任務...")
            while True:
                status_res = requests.get(host + "mission_queue/" + str(mission_id), headers=headers)
                status_data = status_res.json()
                state = status_data.get("state")

                if state == "Done":
                    print("✅ 任務完成:", target_name)
                    break
                elif state == "Failed":
                    print("❌ 任務失敗:", target_name)
                    break
                else:
                    print("⏳ 執行中:", state)
                    time.sleep(2)
        else:
            print(f"❌ 無法送出任務 {target_name}")

    print("\n🏁 所有任務處理完畢。")
def execute_task(object_name):


  
     # 初始化機器手臂和夾爪
    rob, robotiqgrip, joint_acc, joint_vel, tool_acc, tool_vel, joint_tolerance, tool_pose_tolerance = initialize_robot()
    move_joints(rob, [1.571962594985962, -1.5586255232440394, 1.711343765258789, -1.7252147833453577, -1.5701339880572718, -0.002034966145650685], joint_acc, joint_vel, joint_tolerance)
    # 傳送訊息至樹梅派的 IP，同在網域下
    switch_wifi_and_send(
        host='192.168.50.46',
        port=9999,
        ssid='TP-Link_550F',
        message=object_name,
    )
    # 接收從樹梅派回傳的圖像，取得像素座標 (cx, cy) 與 yolo 擷取的物體圖 img
    cx, cy, img = receive_image('0.0.0.0', 9999)
   
    # 機器學習模型將像素座標轉換成實際座標
    real_x_y_position = pixel_to_real_position(cx, cy)
    print(f"{object_name} 的實際位置為: {real_x_y_position}")

    # 強化學習模型預測出控制手臂 wrist3 的角度
    rl_wrist3_radians_angle = rl_main(image=img)

    # #  # 連接手臂wifi TP_Link_550F
    # # reture_original_wifi("TP-Link_550F")
    # # time.sleep(2)

   

    # 手臂轉動強化學習輸出的action角度
    actual_joint_positions, actual_tool_pose = get_position(rob)
    target_joint_configuration = [
        actual_joint_positions[0], actual_joint_positions[1], actual_joint_positions[2],
        actual_joint_positions[3], actual_joint_positions[4], rl_wrist3_radians_angle
    ]
    move_joints(rob, target_joint_configuration, joint_acc, joint_vel, joint_tolerance)

    # 移動至夾取點夾取物體
    robotiqgrip.open_gripper()
    actual_joint_positions, actual_tool_pose = get_position(rob)
    orientation_position = [actual_tool_pose[3], actual_tool_pose[4], actual_tool_pose[5]]
    move_to(rob, [real_x_y_position[0][0], real_x_y_position[0][1], 0.28], orientation_position, tool_acc, tool_vel, tool_pose_tolerance)
    move_to(rob, [real_x_y_position[0][0], real_x_y_position[0][1], 0.26], orientation_position, tool_acc, tool_vel, tool_pose_tolerance)
    # 原本高度0.258/0.22
    robotiqgrip.close_gripper()

    # 手臂拉回到原點
    move_to(rob, [0.10838, -0.48717, 0.45], orientation_position, tool_acc, tool_vel, tool_pose_tolerance)
    # 放置盒子
    move_to(rob, [-0.20960, -0.60555, 0.45], orientation_position, tool_acc, tool_vel, tool_pose_tolerance)
    robotiqgrip.open_gripper()

    # 手臂拉回到原點
    move_joints(rob, [1.571962594985962, -1.5586255232440394, 1.711343765258789, -1.7252147833453577, -1.5701339880572718, -0.002034966145650685], joint_acc, joint_vel, joint_tolerance)

    # 回到原本角度
    actual_joint_positions, actual_tool_pose = get_position(rob)
    target_joint_configuration_ori = [
        actual_joint_positions[0], actual_joint_positions[1], actual_joint_positions[2],
        actual_joint_positions[3], actual_joint_positions[4], math.radians(0)
    ]
    move_joints(rob, target_joint_configuration_ori, joint_acc, joint_vel, joint_tolerance)

    rob.close()  # 結束與機器手臂的連接
    # # 回復原來電腦WiFi
    # reture_original_wifi("AISLab")
    print(f"{object_name} 任務完成")


while True:
    start = input("請輸入數字 1 開始（輸入其他任意鍵退出）：")
    if start == '1':
            while True:
                for task in task_list:
                    execute_task(task)
                print("所有任務執行完畢，準備重新開始。")
    else:
        print("結束程式")
        break
