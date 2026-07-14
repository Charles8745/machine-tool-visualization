import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

def generate_sigmoid_s_curve(V_max=600, V_max1 = 574, A_max1 = 2500,A_max=2311, A_avg=2117,  A_avg1 = 2056, S_max=400, N=200, plot=False):
  

    # === Time segment calculation ===
    Ta = V_max / A_avg
    Tb = (2 * V_max / A_max) - Ta
    Tc = max(0.001, (Ta - Tb) / 2)
    Ts = max(0.001, S_max / V_max - Ta)
    total_time = 2 * Ta + Ts
    
    Ta1 = V_max1 / A_avg1;
    Tb1 = (2 * V_max1 / A_max1) - Ta1;
    Tc1 = max(0.001, (Ta1 - Tb1) / 2);
    Ts1 = max(0.001, S_max / V_max1 - Ta1);

    # Print parameters
    print("==== Sigmoid S-Curve Motion Parameters ====")
    print(f"V_max      = {V_max:.1f} mm/s")
    print(f"A_max      = {A_max:.1f} mm/s²")
    print(f"A_avg      = {A_avg:.1f} mm/s²")
    print(f"S_max      = {S_max:.1f} mm")
    print(f"V_max1      = {V_max1:.1f} mm/s")
    print(f"A_max1      = {A_max1:.1f} mm/s²")
    print(f"A_avg1      = {A_avg1:.1f} mm/s²")
    

    print(f"Total time = {total_time:.4f} s")
    print("===========================================")

    # === Sigmoid parameters ===
    k = 20 / Tc  # control slope

    # === Acceleration profile segments ===
    def sigmoid_section(t, center, scale=1.0):
        s = 1 / (1 + np.exp(-k * (t - center)))
        return (s - s[0]) / (s[-1] - s[0]) * scale

    # t1: rising sigmoid
    t1 = np.linspace(0, Tc, N)
    acc1 = A_max * sigmoid_section(t1, Tc / 2)

    # t2: transition (plateau rising)
    t2 = np.linspace(t1[-1], t1[-1] + Tb, N)
    acc2 = acc1[-1] + (A_max - acc1[-1]) * sigmoid_section(t2, np.mean(t2))

    # t3: falling sigmoid to 0
    t3 = np.linspace(t2[-1], t2[-1] + Tc, N)
    acc3 = A_max * (1 - sigmoid_section(t3, np.mean(t3)))

    # t4: constant velocity segment
    t4 = np.linspace(t3[-1], t3[-1] + Ts1, N)
    acc4 = np.zeros_like(t4)

    # t5: negative rising sigmoid (deceleration)
    t5 = np.linspace(t4[-1], t4[-1] + Tc, N)
    acc5 = -A_max * sigmoid_section(t5, np.mean(t5))

    # t6: negative plateau
    t6 = np.linspace(t5[-1], t5[-1] + Tb, N)
    acc6 = -A_max * np.ones_like(t6)

    # t7: falling sigmoid back to 0
    t7 = np.linspace(t6[-1], t6[-1] + Tc, N)
    acc7 = -A_max * (1 - sigmoid_section(t7, np.mean(t7)))

    # === Concatenate all ===
    t_full = np.concatenate([t1, t2, t3, t4, t5, t6, t7])
    acc_full = np.concatenate([acc1, acc2, acc3, acc4, acc5, acc6, acc7])

    # === Remove duplicates ===
    t_full, idx = np.unique(t_full, return_index=True)
    acc_full = acc_full[idx]

    # === Integrate to get velocity and position ===
    from scipy.integrate import cumtrapz
    vel = cumtrapz(acc_full, t_full, initial=0)
    pos = cumtrapz(vel, t_full, initial=0)
    pos *= S_max / pos[-1]  # rescale to final position

    # === Jerk via spline derivative ===
    cs = CubicSpline(t_full, acc_full)
    jerk = cs.derivative()(t_full)

    # # === Plot ===
    # if plot:
    #     plt.close('all')
    #     plt.figure(figsize=(10, 10))

    #     plt.subplot(3, 1, 1)
    #     plt.plot(t_full, acc_full, 'b', linewidth=2)
    #     plt.title('Acceleration')
    #     plt.ylabel('mm/s²')
    #     plt.grid(True)

    #     plt.subplot(3, 1, 2)
    #     plt.plot(t_full, vel, 'b', linewidth=2)
    #     plt.title('Velocity')
    #     plt.ylabel('mm/s')
    #     plt.grid(True)

    #     plt.subplot(3, 1, 3)
    #     plt.plot(t_full, pos, 'b', linewidth=2)
    #     plt.title('Position')
    #     plt.ylabel('mm')
    #     plt.grid(True)



    #     plt.tight_layout()
    #     plt.show()

    return t_full, acc_full, vel, pos, jerk

# t, acc, vel, pos, jerk = generate_sigmoid_s_curve()