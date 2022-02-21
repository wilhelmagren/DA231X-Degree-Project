import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')

X = pd.read_csv('data/pvt_ind_data.tsv', sep='\t')

x = X['RTrecip']
x = 1 / x

diffs = []
colors = []
labels = []
for idx in range(0, x.shape[0], 2):
    control, psd = x[idx:idx+2]
    diffs.append(100*(psd - control))
    colors.append('lightgreen' if diffs[-1] < 0 else 'indianred')
    labels.append('')

plt.bar(np.arange(len(diffs)), diffs, color=colors)
plt.ylabel('Time [ms]')
plt.title('Reaction time differences between control and sleep deprivation')
plt.xlabel('Subject')
plt.show()
