import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
__root__ = Path(os.getcwd())
labels = Path(__root__, 'data/subjects.tsv')

df = pd.read_csv(labels, sep='\t')

np_gender = np.array(df['Gender'])
np_age = np.array(df['Age'])
df_gender = Counter(np.array(df['Gender']))
df_age = Counter(np.array(df['Age']))

np_gender = [(gender, gender, gender, gender) for gender in np_gender]
np_age = [(age, age, age, age) for age in np_age]
np_gender = [item for sublist in np_gender for item in sublist]
np_age = [item for sublist in np_age for item in sublist]
np_gender = np.array(np_gender)
np_age = np.array(np_age)

females = [idx for idx in range(len(np_gender)) if np_gender[idx] == 'F']
males = [idx for idx in range(len(np_gender)) if np_gender[idx] == 'M']
print(f'num females: {len(females)}')
print(f'num males: {len(males)}')

split = .7
index_female = np.ceil(len(females)*split).astype(int)
index_male = np.ceil(len(males)*split).astype(int)
train = females[:index_female] + males[:index_male]
valid = females[index_female:] + males[index_male:]
print(train)
print(valid)

lentrain = len(train)
lenvalid = len(valid)
print(np_gender[train])
print(np_gender[valid])
print(np_age[train])
print(np_age[valid])

fig, axs = plt.subplots(2, 2)
axs[0, 0].hist(np_gender[train], bins=2, edgecolor='black', linewidth='2.')
axs[0, 0].set_title('train gender')
axs[0, 1].hist(np_gender[valid], bins=2, edgecolor='black', linewidth='2.')
axs[0, 1].set_title('valid gender')
axs[1, 0].hist(np_age[train], bins=10, edgecolor='black', linewidth='2.')
axs[1, 0].set_title('train age')
axs[1, 1].hist(np_age[valid], bins=10, edgecolor='black', linewidth='2.')
axs[1, 1].set_title('valid age')
fig.legend()
plt.show()

total = lentrain + lenvalid
print(f'split: {100*lentrain/total:.1f}/{100*lenvalid/total:.1f}')