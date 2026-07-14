import pickle
import numpy as np

def pixel_to_real_position(cx,cy):
    model_filename='Random Forest_model.pkl'
    # 測試新數據點的部分
    # 載入模型
    with open(model_filename, 'rb') as model_file:
        loaded_model = pickle.load(model_file)

    # 加載標準化器
    with open('scaler_X.pkl', 'rb') as scaler_X_file:
        scaler_X = pickle.load(scaler_X_file)

    with open('scaler_y.pkl', 'rb') as scaler_y_file:
        scaler_y = pickle.load(scaler_y_file)

    # 輸入的 Pixel_X 和 Pixel_Y
    new_pixel_x = np.array([[cx,cy]])  

    # 對像素數據進行標準化
    new_pixel_x_scaled = scaler_X.transform(new_pixel_x)

    # 進行預測
    new_pred_scaled = loaded_model.predict(new_pixel_x_scaled)

    # 反標準化預測出實際座標結果
    real_x_y_position = scaler_y.inverse_transform(new_pred_scaled)
    print(f'真實座標為: {real_x_y_position[0][0],real_x_y_position[0][1]}')

    return real_x_y_position