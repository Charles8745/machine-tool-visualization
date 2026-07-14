import requests
import time
from robot_ur import initialize_robot, move_joints, move_to, get_position
import math
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

# 使用範例
if __name__ == "__main__":
    # # 任務1，,mir移動到要夾取的點
    my_missions = ["test_middle_mir"]  
    run_missions(my_missions)

    # 初始化機器手臂和夾爪
    rob, robotiqgrip, joint_acc, joint_vel, tool_acc, tool_vel, joint_tolerance, tool_pose_tolerance = initialize_robot()

    #home點
    move_to(rob, [0.346,-0.46, 0.44], [0.086,-3.125,-0.009], tool_acc, tool_vel, tool_pose_tolerance)

    #手臂姿態
    actual_joint_positions, actual_tool_pose = get_position(rob)

     #設定 wrist3 的角度
    rl_wrist3_radians_angle=math.radians(-5)

     # 手臂轉動強action角度
    target_joint_configuration = [
        actual_joint_positions[0], actual_joint_positions[1], actual_joint_positions[2],
        actual_joint_positions[3], actual_joint_positions[4], rl_wrist3_radians_angle
    ]
    move_joints(rob, target_joint_configuration, joint_acc, joint_vel, joint_tolerance)



    # 移動至夾取點夾取物體
    actual_joint_positions, actual_tool_pose = get_position(rob)

    #物體位置
    real_x_y_position=[0.37,-0.568]
    orientation_position = [actual_tool_pose[3], actual_tool_pose[4], actual_tool_pose[5]]
    move_to(rob, [real_x_y_position[0], real_x_y_position[1], 0.28], orientation_position, tool_acc, tool_vel, tool_pose_tolerance)
    move_to(rob, [real_x_y_position[0], real_x_y_position[1], 0.22], orientation_position, tool_acc, tool_vel, tool_pose_tolerance)
    robotiqgrip.close_gripper()
     #home點
    move_to(rob, [0.346,-0.46, 0.44], [0.086,-3.125,-0.009], tool_acc, tool_vel, tool_pose_tolerance)
     #手臂姿態
    move_to(rob, [0.38893, 0.0092, 0.542], [2.434,-2.172,0.299], tool_acc, tool_vel, tool_pose_tolerance)

    # # 任務2，前進工具機位置
    my_missions = ["test_machine2_mir"]  
    run_missions(my_missions)
    move_joints(rob, [1.7559006214141846, -1.2100122610675257, 1.616269588470459, -1.9812849203692835, -1.5811308065997522, 0.13651230931282043], joint_acc, joint_vel, joint_tolerance)
    # #打開夾爪
    robotiqgrip.open_gripper()
    move_joints(rob, [3.2329471111297607, -1.3374903837787073, 1.1870627403259277, -1.3777483145343226, -1.525259796773092, 0.12643207609653473], joint_acc, joint_vel, joint_tolerance)
    rob.close()
    
    #任務3，車子回家
    my_missions = ["test_original_mir"]  
    run_missions(my_missions)



