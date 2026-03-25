import re
import sys
import copy
import matplotlib.pyplot as plt


file_num = len(sys.argv)
data_lib = []
pattern = r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"

for i, filepath in enumerate(sys.argv[1:]):
    file = open(filepath, 'r')
    lines = file.readlines()
    current_best = -2000
    data_list = []
    for line in lines:
        # tmp_line = copy.copy(line)
        tmp = line.split(',')
        if 'EDYP' in tmp[-1]:
            matches = re.findall(pattern, line)
            area = float(matches[0]) * 1e6 
            temp = float(matches[1])
            cost = -1* float(matches[-1])
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

plt.plot(x, y[0], linewidth= 3, label='SA-200')

plt.plot(x, y[1], linewidth= 3, label='SCBO-10')
plt.xlabel("Iteration")
plt.ylabel("Best objective value")
plt.title("DSE Convergence Curve under Constraints")
plt.legend()
plt.grid(True)
plt.show()