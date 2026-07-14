import os
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import csv
from PIL import Image
import cv2
import math
from gymnasium import spaces

def calculate_error_metrics_and_plot(csv_file="error_data_predict_ppo_square.csv"):
        """計算 CSV 檔案中的 RMSE、MAE 和 MSE，並繪製圖表"""
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV 檔案 {csv_file} 不存在！")

        predicted_values = []
        actual_values = []

        with open(csv_file, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                predicted_values.append(float(row["Predicted Value"]))
                actual_values.append(float(row["Actual Value"]))
        
        predicted_values = np.array(predicted_values)
        actual_values = np.array(actual_values)
        plot_pred(y=actual_values,y_pred=predicted_values,model_name='PPO')
        plot_residuals(y=actual_values,y_pred=predicted_values,model_name='PPO')


def score_calculation(y, y_pred):
    MAE =  np.round(mean_absolute_error(y, y_pred), 5)
    RMSE = np.round(np.sqrt(mean_squared_error(y, y_pred)), 5)
    R2_Score = np.round(r2_score(y, y_pred), 5)

    # 小數點後三位
    print(f'MAE: {MAE:.5f}')
    print(f'RMSE: {RMSE:.5f}')
    print(f'R2 Score: {R2_Score:.5f}')


    return MAE, RMSE, R2_Score
def plot_pred(y, y_pred, model_name, save_path=None):
    y_pred=np.degrees(y_pred)
    y=np.degrees(y)
    residuals = y_pred - y
    res_abs = np.abs(residuals)
    

    th_1 = 0
    th_2 = 5
    r1_idx = np.where(res_abs <= th_1)
    r2_idx = np.where((res_abs > th_1) & (res_abs <= th_2))
    r3_idx = np.where(res_abs > th_2)
    
    MAE, RMSE, R2_Score = score_calculation(y, y_pred)
        
    fig = plt.figure(figsize=(8.3, 6))
    ax = fig.add_subplot(111)
    
    plt.scatter(y[r1_idx], y_pred[r1_idx], c='royalblue', alpha=0.15, s=40, label=r'|R|$\leq$' + str(th_1))
    plt.scatter(y[r2_idx], y_pred[r2_idx], c='yellowgreen', alpha=0.15, s=40, label=str(th_1) + r'<|R|$\leq$' + str(th_2))
    plt.scatter(y[r3_idx], y_pred[r3_idx], c='orange', alpha=0.15, s=40, label='|R|>' + str(th_2))
    
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=1.5)
    plt.title('Angle' +' Prediction Results '+ '('+(model_name+ ' Algorithm')+')', fontsize=18)
    plt.xlabel('Actual Value', fontsize=16)
    plt.ylabel('Predicted Value', fontsize=16)
    
    info_show = f'MAE: {MAE:.5f}\nRMSE: {RMSE:.5f}\nR2 Score: {R2_Score:.5f}\n'
    plt.text(0.71, 0.05, info_show, ha='left', va='center', transform=ax.transAxes, fontsize=14)
    plt.legend(loc='upper left')
    plt.grid()

    # 保存圖片到指定路徑
    if save_path:
        plt.savefig(save_path)

    plt.show()

def plot_residuals(y, y_pred, model_name, save_path=None):
    y_pred=np.degrees(y_pred)
    y=np.degrees(y)
    residuals = y_pred - y
    res_abs = np.abs(residuals)
    th_1 = 0
    th_2 = 5
    r1_idx = np.where(res_abs <= th_1)
    r2_idx = np.where((res_abs > th_1) & (res_abs <= th_2))
    r3_idx = np.where(res_abs > th_2)
    
    fig = plt.figure(figsize=(8.3, 6))
    ax = fig.add_subplot(111)
    grid = plt.GridSpec(4, 4, wspace=0.5, hspace=0.5)

    main_ax = plt.subplot(grid[0:3, 1:4])
    plt.scatter(y_pred[r1_idx], residuals[r1_idx], c='royalblue', alpha=0.15, s=40, label=r'|R|$\leq$' + str(th_1))
    plt.scatter(y_pred[r2_idx], residuals[r2_idx], c='yellowgreen', alpha=0.15, s=40, label=str(th_1) + r'<|R|$\leq$' + str(th_2))
    plt.scatter(y_pred[r3_idx], residuals[r3_idx], c='orange', alpha=0.15, s=40, label='|R|>' + str(th_2))
    plt.plot([y_pred.min(), y_pred.max()], [0, 0], 'r--', lw=1.5)

    plt.legend(loc='upper left')
    plt.grid()
    plt.title('Residuals for ' + 'Angle ' + '('+(model_name+ ' Algorithm')+')', fontsize=18)
    
    y_hist = plt.subplot(grid[0:3, 0], xticklabels=[], sharey=main_ax)
    plt.hist(residuals, 60, orientation='horizontal', color='g')
    y_hist.invert_xaxis()
    plt.ylabel('Residuals', fontsize=16)
    plt.grid()
    
    x_hist = plt.subplot(grid[3, 1:4], yticklabels=[], sharex=main_ax)
    plt.hist(y_pred, 60, orientation='vertical', color='g')
    x_hist.invert_yaxis()
    plt.xlabel('Predicted Value', fontsize=16)
    plt.grid()  
    
    MAE, RMSE, R2_Score = score_calculation(y, y_pred)
    info_show = f'MAE: {MAE:.5f}\nRMSE: {RMSE:.5f}\nR2 Score: {R2_Score:.5f}\n'
    plt.text(-0.02, 0.05, info_show, ha='left', va='center', transform=ax.transAxes, fontsize=14)
    
    # 保存圖片到指定路徑
    if save_path:
        plt.savefig(save_path)

    plt.show()

def preprocess_image_for_ppo(image_input):
    """
    將 OpenCV (BGR) 圖片轉為 PPO 模型所需格式 (1, 40, 40, 3)，轉為 RGB 並加 batch 維度。
    """
    if isinstance(image_input, np.ndarray):
        if image_input.shape != (40, 40, 3):
            raise ValueError(f"圖片尺寸應為 (40, 40, 3)，但收到的是 {image_input.shape}")

        # BGR 轉 RGB
        img_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)

        # 加 batch 維度，轉為 shape: (1, 40, 40, 3)
        img_array = np.expand_dims(img_rgb, axis=0)
        return img_array
    else:
        raise TypeError("輸入必須為 numpy.ndarray 格式")

# 主程式
def rl_main(image):

    custom_objects = {
    "clip_range": lambda _: 0.2,
    "lr_schedule": lambda _: 3e-4,}
    # 使用模型預測行動
    model = PPO.load("PPO_ur5_checkpoint_90000_steps",custom_objects=custom_objects)

    image_observation=preprocess_image_for_ppo(image)

    obs = image_observation
    
    action, _states = model.predict(obs, deterministic=True)
    
    if action<0:
       print('小於0度')
       action=abs(((action[0]-17)*10))-90
       action=abs(action)
    elif action>0:
        print('大於0度')
        action=abs(((action[0]-17)*10))-90
        action=-1*action
    elif action==0:
        print(f"{action}，action為0度")
        # action=action
        action=90
    
     
    radian_action=math.radians(action)
    
    print("預測動作:", f"{action}度",f"{radian_action}弧度")
    # 去掉 batch 維度
    img_no_batch = obs[0]  # shape: (40, 40, 3)

    # OpenCV 是用 BGR 儲存，但 preprocess_image_for_ppo 裡沒轉 RGB，所以可以直接存
    cv2.imwrite("output.png", img_no_batch)
    print("✅ 圖片已存成 output.png")

    return radian_action

