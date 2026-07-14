import socket
import struct
import time
import urx  # 控制 UR 機器手臂的套件
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper  # 控制 Robotiq 夾爪的模組
import numpy as np
import sys
import math
robot_ip="192.168.50.114"
# 初始化機器手臂和夾爪的函式
def initialize_robot(ip='192.168.50.114', tcp_port=30001):
    """
    初始化 UR 機器手臂和 Robotiq 夾爪

    參數:
        ip (str): 機器手臂的 IP 地址，預設為 '192.168.50.80'
        tcp_port (int): 機器手臂的 TCP 通訊端口，預設為 30001

    回傳:
        rob (urx.Robot): 已初始化的 UR 機器手臂物件
        robotiqgrip (Robotiq_Two_Finger_Gripper): 已初始化的 Robotiq 夾爪物件
        home_joint_config (list): 預設關節配置
        joint_acc (float): 關節加速度
        joint_vel (float): 關節速度
        tool_acc (float): 末端加速度
        tool_vel (float): 末端速度
        joint_tolerance (float): 關節誤差
        tool_pose_tolerance (list): 末端誤差
    """
    # 建立 UR 機器手臂物件並初始化夾爪
    rob = urx.Robot(ip)
    rob.secmon._s_secondary.settimeout(2.0)
    robotiqgrip = Robotiq_Two_Finger_Gripper(rob)

   
    # 設定移動速度與加速度（Joint & Tool）
    joint_acc = 1.4  # 關節加速度
    joint_vel = 1.04  # 關節速度
    tool_acc = 1  # 末端加速度
    tool_vel = 1  # 末端速度

    # 容許誤差（位置與姿態）
    joint_tolerance = 0.01
    tool_pose_tolerance = [0.002, 0.002, 0.002, 0.01, 0.01, 0.01]
    
    return rob, robotiqgrip, joint_acc, joint_vel, tool_acc, tool_vel, joint_tolerance, tool_pose_tolerance

# 啟動夾爪（可初始化夾爪）
def gripper_activate(robotiqgrip):
    urscript = robotiqgrip._get_new_urscript(activate=True)
    robotiqgrip.robot.send_program(urscript)

# 解析從機器手臂回傳的 TCP 狀態資料
def parse_tcp_state_data(state_data, subpackage):
    data_bytes = bytearray()
    data_bytes.extend(state_data)
    data_length = struct.unpack("!i", data_bytes[0:4])[0]
    robot_message_type = data_bytes[4]
    assert(robot_message_type == 16)
    byte_idx = 5

    # 定義子封包類型
    subpackage_types = {'joint_data': 1, 'cartesian_info': 4, 'force_mode_data': 7, 'tool_data': 2}

    while byte_idx < data_length:
        package_length = struct.unpack("!i", data_bytes[byte_idx:(byte_idx+4)])[0]
        byte_idx += 4
        package_idx = data_bytes[byte_idx]
        if package_idx == subpackage_types[subpackage]:
            byte_idx += 1
            break
        byte_idx += package_length - 4

    # 解析關節位置
    def parse_joint_data(data_bytes, byte_idx):
        actual_joint_positions = [0]*6
        target_joint_positions = [0]*6
        for joint_idx in range(6):
            actual_joint_positions[joint_idx] = struct.unpack('!d', data_bytes[(byte_idx+0):(byte_idx+8)])[0]
            target_joint_positions[joint_idx] = struct.unpack('!d', data_bytes[(byte_idx+8):(byte_idx+16)])[0]
            byte_idx += 41
        return actual_joint_positions

    # 解析末端位置與姿態
    def parse_cartesian_info(data_bytes, byte_idx):
        actual_tool_pose = [0]*6
        for pose_value_idx in range(6):
            actual_tool_pose[pose_value_idx] = struct.unpack('!d', data_bytes[(byte_idx+0):(byte_idx+8)])[0]
            byte_idx += 8
        return actual_tool_pose

    # 解析夾爪的類比輸入資料
    def parse_tool_data(data_bytes, byte_idx):
        byte_idx += 2
        tool_analog_input2 = struct.unpack('!d', data_bytes[(byte_idx+0):(byte_idx+8)])[0]
        return tool_analog_input2

    parse_functions = {
        'joint_data': parse_joint_data,
        'cartesian_info': parse_cartesian_info,
        'tool_data': parse_tool_data
    }
    return parse_functions[subpackage](data_bytes, byte_idx)

# 移動至指定的關節角度
def move_joints(rob, joint_configuration, joint_acc, joint_vel, joint_tolerance):
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_socket.connect((robot_ip, 30001))  # 使用 robot_ip 來替代硬編碼的 ip
    tcp_command = "movej([%f" % joint_configuration[0]
    for joint_idx in range(1, 6):
        tcp_command += ",%f" % joint_configuration[joint_idx]
    tcp_command += "],a=%f,v=%f)\n" % (joint_acc, joint_vel)
    tcp_socket.send(str.encode(tcp_command))

    # 等待機器手臂到達指定角度
    actual_joint_positions, _ = get_position(rob)
    while not all([np.abs(actual_joint_positions[j] - joint_configuration[j]) < joint_tolerance for j in range(6)]):
        actual_joint_positions, _ = get_position(rob)
        time.sleep(0.01)

    tcp_socket.close()

# 移動至指定的位置與姿態（tool space）
def move_to(rob, tool_position, tool_orientation, tool_acc, tool_vel, tool_pose_tolerance):
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_socket.connect((robot_ip, 30001))
    tcp_command = "movel(p[%f,%f,%f,%f,%f,%f],a=%f,v=%f,t=0,r=0)\n" % (
        tool_position[0], tool_position[1], tool_position[2],
        tool_orientation[0], tool_orientation[1], tool_orientation[2],
        tool_acc, tool_vel
    )
    tcp_socket.send(str.encode(tcp_command))

    # 等待機器手臂到達指定位置
    _, actual_tool_pose = get_position(rob)
    while not all([np.abs(actual_tool_pose[j] - tool_position[j]) < tool_pose_tolerance[j] for j in range(3)]):
        _, actual_tool_pose = get_position(rob)
    tcp_socket.close()

# 回到預設 Home 位置
def go_home(rob, home_joint_config):
    move_joints(rob, home_joint_config, 1.4, 1.05, 0.01)

# 執行夾取任務（夾爪移至目標點、關閉、再抬起）
def grasp(robotiqgrip, position, angle):
    position_above = [position[0], position[1], 0.135]  # 目標上方一段距離
    orientation = [np.cos(-np.deg2rad(angle)/2)*np.pi, np.sin(-np.deg2rad(angle)/2)*np.pi, 0]
    move_to(robotiqgrip.robot, position_above, orientation, 0.2, 0.2, [0.002, 0.002, 0.002, 0.01, 0.01, 0.01])
    move_to(robotiqgrip.robot, position, orientation, 0.2, 0.2, [0.002, 0.002, 0.002, 0.01, 0.01, 0.01])
    robotiqgrip.close_gripper()  # 關閉夾爪
    move_to(robotiqgrip.robot, position_above, orientation, 0.2, 0.2, [0.002, 0.002, 0.002, 0.01, 0.01, 0.01])


def get_position(rob):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as tcp_socket:
            tcp_socket.settimeout(2.0)  # 設定 2 秒 timeout，防止 recv 卡住
            tcp_socket.connect((robot_ip, 30002))

            max_attempts = 10  # 最多讀取 10 次
            for _ in range(max_attempts):
                try:
                    msg = tcp_socket.recv(2048)
                    msgHEX = msg.hex()
                    msgType = msgHEX[8:10]

                    if int(msgType, 16) == 20:
                        if int(msgHEX[28:30], 16) == 6:
                            print(f"Error code: C{int(msgHEX[30:38], 16)} A{int(msgHEX[38:46], 16)}")
                            return None, None

                    elif int(msgType, 16) == 16:
                        actual_joint_positions = parse_tcp_state_data(msg, 'joint_data')
                        actual_tool_pose = parse_tcp_state_data(msg, 'cartesian_info')
                        print('actual_tool_pose', actual_tool_pose, '\nactual_joint_positions', actual_joint_positions)
                        return actual_joint_positions, actual_tool_pose
                    
                    time.sleep(0.01)

                except socket.timeout:
                    print("⚠️ 資料接收超時，重試中...")
                    continue
                except Exception as e:
                    print(f"❌ 接收過程發生錯誤: {e}")
                    return None, None

    except Exception as conn_err:
        print(f"❌ 無法建立與機器人的連線：{conn_err}")
        return None, None

    print("⚠️ 未取得有效數據")
    return None, None


# #程式
if __name__ == "__main__":
    # 初始化機器手臂和夾爪
    rob, robotiqgrip, joint_acc, joint_vel, tool_acc, tool_vel, joint_tolerance, tool_pose_tolerance = initialize_robot()
    # 關閉夾爪
    robotiqgrip.close_gripper()

    
    
    # 逆運動學移動至某點、開關夾爪
    actual_joint_positions, actual_tool_pose = get_position(rob)
#     move_to(rob, [0.58, -0.01, 0.5], [3.14, 0, 0], tool_acc, tool_vel, tool_pose_tolerance)

#     # # # 關節配置
#     # # target_joint_configuration = [0.5, -1.2, 1.0, -1.0, 1.5, 0.0]  #根據需要更改
#     # # # 呼叫 move_joints 函數，進行關節移動
#     # # move_joints(rob, target_joint_configuration, joint_acc, joint_vel, joint_tolerance)

#     # # 開夾爪
#     robotiqgrip.open_gripper()

#     rob.close()  # 結束與機器手臂的連接
#     # print("任務完成，程式結束。")
 
