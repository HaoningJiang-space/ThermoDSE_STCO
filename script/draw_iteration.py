import re
import sys
import copy
import matplotlib.pyplot as plt
import seaborn as sns

file_num = len(sys.argv)
data_lib = []
pattern = r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"

for i, filepath in enumerate(sys.argv[1:]):
    file = open(filepath, 'r')
    lines = file.readlines()
    current_best = -500
    data_list = []
    for line in lines:
        # tmp_line = copy.copy(line)
        tmp = line.split(',')
        if 'EDYP' in tmp[-1]:
            matches = re.findall(pattern, line)
            area = float(matches[0]) * 1e6 
            temp = float(matches[1])
            cost = -1* float(matches[-1]) / 1.8
            cost = round(cost, 1)
            if area < 300 and temp < 348 and cost > current_best:
                current_best = cost 
            data_list.append(current_best)
    data_lib.append(data_list)

max_iteration = 0

for data_list in data_lib:
    iteration = len(data_list)
    if iteration > max_iteration:
        max_iteration = iteration
y = []
for data_list in data_lib:
    data = data_list[-1]
    print(len(data_list))
    data_list = data_list + [data ]* (max_iteration - len(data_list))
    print(len(data_list))
    y.append(data_list)

x = [a for a in range(1,max_iteration+1)]

max_cost4, cov4= max(y[0]),y[0].index(max(y[0]))
max_cost10, cov10= max(y[1]),y[1].index(max(y[1]))

max_cost16, cov16= max(y[2]),y[2].index(max(y[2]))


print(f'th=4, min cost:{max(y[0])}, coverage at:{y[0].index(max(y[0]))}')
print(f'th=10, min cost:{max(y[1])}, coverage at:{y[1].index(max(y[1]))}')
print(f'th=16, min cost:{max(y[2])}, coverage at:{y[2].index(max(y[2]))}')

plt.figure(figsize=( 5,3), dpi=300)
plt.ylim(-200, -120)
plt.plot(x, y[0], linewidth= 2, label='SCBO-4')

plt.plot(x, y[1], linewidth= 2, label='SCBO-10')

plt.plot(x, y[2], linewidth= 2, label='SCBO-16')

plt.scatter(cov4,max_cost4, s=75, color='blue', marker='*', edgecolors='blue',
            label=r'$\mathit{th}=4$', zorder=5)

plt.scatter(cov10,max_cost10, s=75, color='darkorange', marker='*', edgecolors='darkorange',
            label=r'$\mathit{th}=10$', zorder=5)

plt.scatter(cov16,max_cost16, s=75, color='green', marker='*', edgecolors='green',
            label=r'$\mathit{th}=16$', zorder=5)

plt.text(cov4-50, max_cost4+7, f'({cov4}, {max_cost4})', fontsize=8, color='blue',  ha='center', va='top')
plt.text(cov10+40, max_cost10+7, f'({cov10}, {max_cost10})', fontsize=8, color='darkorange',  ha='center', va='top')
plt.text(cov16, max_cost16+5, f'({cov16}, {max_cost16})', fontsize=8, color='green',  ha='center', va='top')


plt.xlabel("Iteration")
plt.ylabel("Current Cost")
# plt.title("Convergence Curve under Constraints")
plt.legend()
plt.tight_layout()
# plt.grid(True)
plt.savefig('scbo_coverg.png')
plt.show()
