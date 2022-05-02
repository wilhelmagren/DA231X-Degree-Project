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
indices_female = np.arange(len(females))
indices_male = np.arange(len(males))
print(indices_female)
np.random.shuffle(indices_female)
np.random.shuffle(indices_male)
females = np.array(females)
males = np.array(males)
females = females[indices_female]
males = males[indices_male]
train = np.concatenate((females[:index_female], males[:index_male]))
valid = np.concatenate((females[index_female:], males[index_male:]))
print('training genders')
print(len(females[:index_female]))
print(len(males[:index_male]))

print('validation genders')
print(len(females[index_female:]))
print(len(males[index_male:]))

print(train)
print(valid)

#train = [73, 114, 38, 14, 48, 125, 113, 81, 15, 12, 122, 119, 68, 124, 78, 79, 49, 110, 59, 39, 118, 77, 62, 75, 3, 51, 61 ,130,  65, 106,  72,  32, 117,  83 ,127, 60,  66,  50, 111,  27,  26, 104,  70,  37,  63,  82, 115,  98,  90,  88, 108,   2,  25, 96,  58,  64,  13,  80,  36,  89,  34, 105, 126,  91, 128,  57,  56, 121,  33,  99,  30, 85,  43,   7, 102,  42,  47,   9,  10,  45,  53,  29,  52,   4,  54,   6,  46,  92,  11, 87,  22,  23, 100, 103,  17,  84,  95,  41,   8,  19,  40,   5,  20,  93,  18,  31]
#valid = [107, 112,  97, 123,  24,   1,  74,  35,  69,  76, 120,   0,  71, 109, 116, 129,  67,  28, 44,  16, 101,  21,  55,  86,  94]

lentrain = len(train)
lenvalid = len(valid)
print(np_gender[train])
print(np_gender[valid])
print(np_age[train])
print(np_age[valid])

print('training set stats')
print(np_age[train].mean())
print(np_age[train].std())
print(np_age[train].min())
print(np_age[train].max())
print('validation set stats')
print(np_age[valid].mean())
print(np_age[valid].std())
print(np_age[valid].min())
print(np_age[valid].max())


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

print(lentrain)
print(lenvalid)