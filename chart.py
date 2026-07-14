import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def normalize_time(t):
    t = t.strip()
    if 'T' not in t and len(t) == 14:
        return t[:8] + 'T' + t[8:]  # Add 'T' between date and time
    return t

def closest_time(time_series, target_time):
    """
    Find the closest time in time_series to the target_time.
    """
    # Adjusting format to handle the 'T' in the time strings
    time_series = pd.to_datetime(time_series, format='%Y%m%dT%H%M%S')
    target_time = pd.to_datetime(target_time, format='%Y%m%dT%H%M%S')

    # Find the closest time
    closest_idx = (time_series - target_time).abs().argmin()
    return time_series.iloc[closest_idx]

def plot_data(start_time="202505021145", end_time="202505021713"):
    plt.close('all') 
    start_time = normalize_time(start_time + "00")  # Add '00' to make it seconds
    end_time = normalize_time(end_time + "00")  # Add '00' to make it seconds

    df = pd.read_excel('./DrillingData.xlsx')
    time_labels = df.iloc[:, 0].astype(str).str.strip()  # Clean up time column

    # Normalize time labels to match the format
    df[0] = time_labels

    # Find the closest time in the dataframe to the given start_time and end_time
    closest_start_time = closest_time(time_labels, start_time)
    closest_end_time = closest_time(time_labels, end_time)

    print(f"Closest start time: {closest_start_time}")
    print(f"Closest end time: {closest_end_time}")

    # Filter data based on the closest times
    df_filtered = df[(time_labels >= closest_start_time.strftime('%Y%m%dT%H%M%S')) & 
                     (time_labels <= closest_end_time.strftime('%Y%m%dT%H%M%S'))]
    time_filtered = df_filtered.iloc[:, 0]

    if df_filtered.empty:
        print("指定的時間區間內沒有資料。")
        return

    # First plot
    plt.figure()
    plt.plot(time_filtered, df_filtered.iloc[:, 2])
    plt.xticks([time_filtered.iloc[0], time_filtered.iloc[-1]])
    plt.xlabel('Time')
    plt.ylabel('Spindle torque(%)')
    plt.title(f'Time vs Spindle torque\n({closest_start_time} to {closest_end_time})')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Second plot
    plt.figure()
    plt.plot(time_filtered, df_filtered.iloc[:, 3], label='X axis')
    plt.plot(time_filtered, df_filtered.iloc[:, 4], label='Y axis')
    plt.plot(time_filtered, df_filtered.iloc[:, 5], label='Z axis')
    plt.xticks([time_filtered.iloc[0], time_filtered.iloc[-1]])
    plt.xlabel('Time')
    plt.ylabel('Vibration(g-rms)')
    plt.title(f'Time vs Vibration - X,Y,Z axis(g-rms)\n({closest_start_time} to {closest_end_time})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_data()
