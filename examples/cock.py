from scipy.stats import wilcoxon

normal = [70.9, 49.0, 83.2, 73.3, 81.4, 68.6, 
1.362, 2.846, 0.120, 0.262, 0.141, 0.269]
alternative = [67.5, 65.7, 82.6, 80.1, 79.9, 78.2,
1.848, 2.082, 0.150, 0.171, 0.152, 0.173]

w, p = wilcoxon(normal, y=alternative)

print(f'{w=:.4f}, {p=:.4f}')