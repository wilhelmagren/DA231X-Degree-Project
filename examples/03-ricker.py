import matplotlib.pyplot as plt
from scipy.signal import ricker

vec = ricker(100, 4)

plt.title('Ricker wavelet')
plt.plot(list(range(-50, 50)), vec)
plt.show()
