import matplotlib.pyplot as plt

# 数据
labels = ['Comp', 'SRAM', 'Other', 'ICS']
sizes = [49.2, 14.2, 19.0, 17.6]
colors = ['#4c72b0', '#dd8452', '#55a868', '#8172b2']  # 可自定义配色
explode = [0.0, 0, 0, 0]  # 让第一个稍突出

# 绘图
fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
wedges, texts, autotexts = ax.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%',
    startangle=90,
    colors=colors,
    explode=explode,
    textprops={'fontsize': 16}
)

# 设置等宽比例保证是圆形
ax.axis('equal')

# plt.title('Chip Area Composition', fontsize=14)
plt.tight_layout()
plt.savefig('area_bkd.png')
plt.show()