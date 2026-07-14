import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import pickle
import xgboost as xgb  # 引入 XGBoost
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from matplotlib.gridspec import GridSpec

def score_calculation(y, y_pred):
    MAE = np.round(mean_absolute_error(y, y_pred), 5)
    RMSE = np.round(np.sqrt(mean_squared_error(y, y_pred)), 5)
    R2_Score = np.round(r2_score(y, y_pred), 5)

    # 小數點後三位
    print(f'MAE: {MAE:.5f}')
    print(f'RMSE: {RMSE:.5f}')
    print(f'R2 Score: {R2_Score:.5f}')

    return MAE, RMSE, R2_Score
    
def plot_pred(y, y_pred, model_name, position_name, save_path=None):
    residuals = y_pred - y
    res_abs = np.abs(residuals)
    th_1 = 0.0008
    th_2 = 0.0012
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
    plt.title(model_name + ' Prediction Results' + f" ({position_name})", fontsize=18)
    plt.xlabel('Actual Value', fontsize=16)
    plt.ylabel('Predicted Value', fontsize=16)
    
    info_show = f'MAE: {MAE:.5f}\nRMSE: {RMSE:.5f}\nR2 Score: {R2_Score:.5f}\n'
    plt.text(0.68, 0.05, info_show, ha='left', va='center', transform=ax.transAxes, fontsize=16)
    plt.legend(loc='upper left')
    plt.grid()

    # 保存圖片到指定路徑
    if save_path:
        plt.savefig(save_path)

    plt.show()

def plot_residuals(y, y_pred, model_name, position_name, save_path=None):
    # 計算殘差
    residuals = y_pred - y
    res_abs = np.abs(residuals)
    th_1 = 0.0008
    th_2 = 0.0012
    r1_idx = np.where(res_abs <= th_1)
    r2_idx = np.where((res_abs > th_1) & (res_abs <= th_2))
    r3_idx = np.where(res_abs > th_2)

    # 創建主圖與子圖的佈局
    fig = plt.figure(figsize=(9, 6))
    grid = GridSpec(4, 4, wspace=0.5, hspace=0.5)

    # 主圖 (殘差分佈)
    main_ax = fig.add_subplot(grid[0:3, 1:4])
    main_ax.scatter(y_pred[r1_idx], residuals[r1_idx], c='royalblue', alpha=0.15, s=40, label=r'|R|$\leq$' + str(th_1))
    main_ax.scatter(y_pred[r2_idx], residuals[r2_idx], c='yellowgreen', alpha=0.15, s=40, label=str(th_1) + r'<|R|$\leq$' + str(th_2))
    main_ax.scatter(y_pred[r3_idx], residuals[r3_idx], c='orange', alpha=0.15, s=40, label='|R|>' + str(th_2))
    main_ax.plot([y_pred.min(), y_pred.max()], [0, 0], 'r--', lw=1.5)

    main_ax.legend(loc='upper left')
    main_ax.set_title('Residuals for ' + model_name + f" ({position_name})", fontsize=18)
    main_ax.grid()

    # 殘差直方圖 (左側)
    y_hist = fig.add_subplot(grid[0:3, 0], sharey=main_ax)
    y_hist.hist(residuals, bins=60, orientation='horizontal', color='g')
    y_hist.invert_xaxis()
    y_hist.set_ylabel('Residuals', fontsize=16)
    y_hist.xaxis.set_visible(False)

    y_hist.grid()

    # 預測值直方圖 (底部)
    x_hist = fig.add_subplot(grid[3, 1:4], sharex=main_ax)
    x_hist.hist(y_pred, bins=60, orientation='vertical', color='g')
    x_hist.invert_yaxis()
    x_hist.set_xlabel('Predicted Value', fontsize=16)
    x_hist.yaxis.set_visible(False)

    x_hist.grid()

    # 計算指標並顯示文字資訊
    MAE, RMSE, R2_Score = score_calculation(y, y_pred)
    info_show = f'MAE: {MAE:.5f}\nRMSE: {RMSE:.5f}\nR2 Score: {R2_Score:.5f}'
    
    plt.text(-0.42, -0.15, info_show, transform=main_ax.transAxes, ha='left', va='top', fontsize=16)
    
    # 保存圖片到指定路徑
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()


# Load CSV data (更新文件路徑)
data = pd.read_csv('corners_data22.csv')

# 提取輸入特徵 (Pixel_X, Pixel_Y) 和輸出標籤 (World_X, World_Y)
X = data[['Pixel_X', 'Pixel_Y']].values  # 特徵
y = data[['World_X', 'World_Y']].values  # 目標

# 特徵 MinMaxScaler 標準化
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)  # 將數據縮放到 [0, 1]

# 目標標籤 MinMaxScaler 標準化
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y)  # 將數據縮放到 [0, 1]

# 保存 MinMaxScaler 標準化器
with open('scaler_X.pkl', 'wb') as scaler_X_file:
    pickle.dump(scaler_X, scaler_X_file)

with open('scaler_y.pkl', 'wb') as scaler_y_file:
    pickle.dump(scaler_y, scaler_y_file)

# 定義可用的回歸模型
models = {
    'SVR': MultiOutputRegressor(SVR()),
    'Random Forest': MultiOutputRegressor(RandomForestRegressor(n_jobs=-1)),  # 利用所有 CPU 核心
    'XGBoost': MultiOutputRegressor(xgb.XGBRegressor(n_jobs=-1)),  # XGBoost 也支持並行
    'KNeighbors': MultiOutputRegressor(KNeighborsRegressor()),
    'MLP': MultiOutputRegressor(MLPRegressor()), 
}


# 選擇要使用的模型
selected_model_name = 'Random Forest'  # 更改為 'SVR', 'Random Forest', 'KNeighbors', 'XGBoost','MLP' 
selected_model = models[selected_model_name]

# 初始化 K-Fold 交叉驗證
kf = KFold(n_splits=10, shuffle=True)
y_true_all = []
y_pred_all = []

# 進行 K-Fold 交叉驗證
for train_index, test_index in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y_scaled[train_index], y_scaled[test_index]
    
    # 訓練模型
    selected_model.fit(X_train, y_train)
    
    # 進行預測
    y_pred = selected_model.predict(X_test)
    
    # 收集結果
    y_true_all.append(scaler_y.inverse_transform(y_test))  # 反標準化 y
    y_pred_all.append(scaler_y.inverse_transform(y_pred))  # 反標準化預測結果

# 將列表轉換為 NumPy 陣列
y_true_all = np.vstack(y_true_all)  # 垂直堆疊真實值
y_pred_all = np.vstack(y_pred_all)  # 垂直堆疊預測值

# 保存模型
model_filename = f'{selected_model_name}_model.pkl'
with open(model_filename, 'wb') as model_file:
    pickle.dump(selected_model, model_file)
print(f'Model saved as {model_filename}')

# # 動態生成保存路徑
# x_pred_save_path = f'{selected_model_name}_world_x_prediction.png'
# x_residual_save_path = f'{selected_model_name}_world_x_residuals.png'
# y_pred_save_path = f'{selected_model_name}_world_y_prediction.png'
# y_residual_save_path = f'{selected_model_name}_world_y_residuals.png'

# # 繪製並保存 X 軸預測結果圖像
# plot_pred(y_true_all[:, 0], y_pred_all[:, 0], selected_model_name ,position_name='X Position', save_path=x_pred_save_path)
# plot_residuals(y_true_all[:, 0], y_pred_all[:, 0], selected_model_name ,position_name='X Position', save_path=x_residual_save_path)

# # 繪製並保存 Y 軸預測結果圖像
# plot_pred(y_true_all[:, 1], y_pred_all[:, 1], selected_model_name ,position_name='Y Position', save_path=y_pred_save_path)
# plot_residuals(y_true_all[:, 1], y_pred_all[:, 1], selected_model_name ,position_name='Y Position', save_path=y_residual_save_path)

# print(f'Saved X axis prediction plot as {x_pred_save_path}')
# print(f'Saved X axis residuals plot as {x_residual_save_path}')
# print(f'Saved Y axis prediction plot as {y_pred_save_path}')
# print(f'Saved Y axis residuals plot as {y_residual_save_path}')


# # 測試新數據點的部分
# # 載入模型
# with open(model_filename, 'rb') as model_file:
#     loaded_model = pickle.load(model_file)

# # 加載標準化器
# with open('scaler_X.pkl', 'rb') as scaler_X_file:
#     scaler_X = pickle.load(scaler_X_file)

# with open('scaler_y.pkl', 'rb') as scaler_y_file:
#     scaler_y = pickle.load(scaler_y_file)

# # 假設我們要測試的 Pixel_X 和 Pixel_Y
# new_pixel_x = np.array([[200, 168.77245]])  # 替換為你的實際值

# # 對新數據進行標準化
# new_pixel_x_scaled = scaler_X.transform(new_pixel_x)

# # 進行預測
# new_pred_scaled = loaded_model.predict(new_pixel_x_scaled)

# # 反標準化預測結果
# new_pred = scaler_y.inverse_transform(new_pred_scaled)

# print(f'Predicted World Coordinates for Pixel ({new_pixel_x[0][0]}, {new_pixel_x[0][1]}): {new_pred[0]}')
