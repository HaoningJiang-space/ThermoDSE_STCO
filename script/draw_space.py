import re
import sys
import copy
import matplotlib.pyplot as plt
import numpy as np

def remove_duplicate_close_points_anisotropic(x, y, delta_x=0.5, delta_y=2.0):
    xy = np.column_stack((x, y))
    xy = xy[np.lexsort((xy[:,1], xy[:,0]))]
    noise= np.random.uniform(-0.2,0.2)
    filtered = [xy[0]]
    for i in range(1, len(xy)):
        dx = abs(xy[i,0] - filtered[-1][0])
        dy = abs(xy[i,1] - filtered[-1][1])
        if dx > delta_x * (1+ noise) or dy > delta_y *(1+ noise):
            filtered.append(xy[i])
    return np.array(filtered).T


# -----------------------------
# 计算 Pareto 前沿
# 目标：minimize both EDYP and Peak_Temp
# -----------------------------
def pareto_frontier(x, y):
    """
    计算二维 Pareto 前沿（都越小越好）
    返回前沿点 (x_sorted, y_pareto)
    """
    sorted_idx = np.argsort(x)
    x_sorted = x[sorted_idx]
    y_sorted = y[sorted_idx]
    pareto_x, pareto_y = [x_sorted[0]], [y_sorted[0]]
    min_y = y_sorted[0]
    for i in range(1, len(x_sorted)):
        if y_sorted[i] < min_y:
            pareto_x.append(x_sorted[i])
            pareto_y.append(y_sorted[i])
            min_y = y_sorted[i]
    return np.array(pareto_x), np.array(pareto_y)


filepath = sys.argv[1]
ile = open(filepath, 'r')

edyp = []
temp = []
with open(filepath, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if 'sys_info' in line:
            pattern = r"(\w+):\s*\(?([^,()]+(?:,[^A-Za-z()]+)*)"
            pairs = re.findall(pattern, line)
            item = {}
            # print(pairs)
            for key, val in pairs:
                nums = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+", val)
                # item[key] = list(map(float, nums))
                # print(key, val, nums)
                if key == 'peak_temp':
                    temp_value = float(nums[0])-273
                if key == 'EDYP':
                    cost_value = round(float(nums[0]),3)
                if key == 'Area':
                    area_value = float(nums[0])
            if area_value < 0.0003:
                temp.append(temp_value)
                edyp.append(cost_value)
# print(edyp[0:100])
# print(temp[0:100])

print(f'Best EDYP:{min(edyp)}, its peak temp:{temp[edyp.index(min(edyp))]}')

edyp, temp = np.array(edyp), np.array(temp)

mask_low = temp <= 75
mask_high = temp > 75
edyp_low_min  = np.min(edyp[mask_low])  if np.any(mask_low)  else np.nan
edyp_high_min = np.min(edyp[mask_high]) if np.any(mask_high) else np.nan

idx_low  = np.argmin(np.where(mask_low,  edyp, np.inf))
idx_high = np.argmin(np.where(mask_high, edyp, np.inf))

t_low, e_low   = temp[idx_low], edyp[idx_low]
t_high, e_high = temp[idx_high], edyp[idx_high]

print(f"feasible minimum(Temp ≤ 75):  EDYP = {e_low:.3f} (Temp={t_low:.1f})")
print(f"Globle minimum (Temp >  75):  EDYP = {e_high:.3f} (Temp={t_high:.1f})")

temp, edyp = remove_duplicate_close_points_anisotropic(temp, edyp,delta_x=1, delta_y=7)
px, py = pareto_frontier(temp, edyp)



# --- 绘制散点图 ---
plt.figure(figsize=(6,4), dpi=300)
# plt.legend(loc='upper right')
plt.scatter( temp, edyp,  marker='x', color='black', alpha=0.3, s=30, linewidths=0.8, label='Design Points')

plt.plot(px, py, color='black', linewidth=1.2, label='Pareto Front')
plt.scatter(px, py, color='white', edgecolors='black', s=35, zorder=3)

# 底线（最小温度参考线）
xmin = 75
plt.axvline(x=xmin, color='Red', linestyle='--', linewidth=0.8)
plt.text(xmin + 1, max(edyp) , 'Temp. Const.', color='Red',
         fontsize=9, ha='left', va='center')


plt.scatter(t_low, e_low, s=70, color='green', marker='*', edgecolors='green',
            label='Feasible Min', zorder=10)
plt.scatter(t_high, e_high, s=70, color='red', marker='8', edgecolors='red', linewidth=1.2,
            label='Over-limit Min ', zorder=10)
plt.text(t_low-3, e_low-2, f'({round(t_low,1)}, {round(e_low,1)})', fontsize=9, color='green',  ha='center', va='top', )
plt.text(t_high+2, e_high-3, f'({round(t_high,1)}, {round(e_high,1)})', fontsize=9, color='red', ha='center', va='top',)
# 坐标轴与标题
plt.xlabel("Peak Temperature (°C)", fontsize=11)
plt.ylabel("EDYP Cost", fontsize=11)
# plt.title("Pareto Front: Peak Temperature vs EDYP", fontsize=12, pad=10)
plt.grid(True, linestyle=':', linewidth=0.5, color='gray', alpha=0.4)
plt.legend(frameon=False, fontsize=8)

plt.tight_layout()
plt.savefig('pareto.png')
plt.show()

