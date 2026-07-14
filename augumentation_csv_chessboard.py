import pandas as pd
import numpy as np

# 讀取 CSV 文件
input_file = 'chessboard_corners.csv'  # 原始文件名
output_file = 'corners_data11.csv'  # 保存結果的文件名
final_output_file = 'corners_data22.csv'  # 最終結果文件名

# 定義插值段數
num_segments = 10 # 根據需要更改這個值

try:
    # 讀取 CSV 數據，保留標題行
    data = pd.read_csv(input_file, header=0)  # header=0 表示第一行是標題

    # 檢查原始數據中是否有缺失值
    if data.isnull().any().any():
        raise ValueError("原始數據中存在缺失值，請檢查文件。")

    # 確保 CSV 文件包含正確的列數
    if data.shape[1] != 4:
        raise ValueError("CSV 文件應該包含 4 列數據（Pixel_X, Pixel_Y, World_X, World_Y）")

    # 創建一個新的列表以存儲插值結果
    combined_rows = []

    # 每兩行進行處理以生成插值
    for i in range(len(data) - 1):  # 修改為處理到倒數第二行
        # 取兩行數據
        row1 = data.iloc[i].values
        row2 = data.iloc[i + 1].values

        # 添加原始的 row1
        combined_rows.append(row1)

        # 計算 Pixel_X 的絕對差值
        pixel_x_difference = abs(row2[0] - row1[0])  # row1[0] 和 row2[0] 是 Pixel_X 的值

        # 根據 Pixel_X 的絕對差值判斷是否生成插值
        if pixel_x_difference <= 200:
            # 根據 row1 和 row2 生成指定的段數 (插值)，所以總共會有 num_segments + 1 個點
            for j in range(1, num_segments + 1):  # 生成 num_segments 段
                # 計算插值
                segment = row1 + (row2 - row1) * (j / (num_segments + 1))  # 插值公式，分成 num_segments + 1 份
                combined_rows.append(segment)

        # 添加原始的 row2
        combined_rows.append(row2)

    # 創建新的 DataFrame
    result_df = pd.DataFrame(combined_rows, columns=data.columns)

    # 去除重複的行
    result_df = result_df.drop_duplicates()

    # 檢查插值結果中是否有缺失值
    if result_df.isnull().any().any():
        raise ValueError("插值結果中存在缺失值。")

    # 將插值結果寫入新的 CSV 文件
    result_df.to_csv(output_file, index=False)
    print(f"插值結果已保存至 {output_file}")

    # 開始搜尋相同的 World_X 並生成新插值
    final_rows = []

    for index, row in result_df.iterrows():
        world_x = row['World_X']
        world_y = row['World_Y']
        # 添加原始的行 (row1)
        final_rows.append(row.values)

        # 找到從當前行以下所有與當前行相同的 World_X 的行
        matching_rows = result_df.iloc[index + 1:]  # 只考慮當前行之下的行
        matching_rows = matching_rows[matching_rows['World_X'] == world_x]

        if not matching_rows.empty:
            # 計算當前行與每個匹配行的 Y 距離（使用 World_Y）
            distances = np.abs(matching_rows['World_Y'] - world_y)
            closest_index = distances.idxmin()  # 找到最近的 World_Y 的索引
            closest_row = matching_rows.loc[closest_index]

            # 根據最近的行生成插值
            for j in range(1, num_segments + 1):
                # 計算插值
                new_segment = row + (closest_row.values - row.values) * (j / (num_segments + 1))
                final_rows.append(new_segment)

        # 添加原始的行
        final_rows.append(row.values)

    # 創建最終的 DataFrame
    final_result_df = pd.DataFrame(final_rows, columns=result_df.columns)

    # 去除重複的行
    final_result_df = final_result_df.drop_duplicates()

    # 清理最終數據框，去除缺失值
    final_result_df = final_result_df.dropna()

    # 將最終結果寫入 CSV 文件
    final_result_df.to_csv(final_output_file, index=False)

    print(f"最終結果已保存至 {final_output_file}")

except FileNotFoundError:
    print(f"文件 {input_file} 未找到，請確認文件路徑是否正確。")
except ValueError as ve:
    print(ve)
except Exception as e:
    print(f"發生錯誤：{e}")
