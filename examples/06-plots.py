import matplotlib.pyplot as plt
from scipy.signal import ricker
import scipy.signal as sig

plt.rc('font', size=18)
plt.rc('axes', titlesize=18)
vec = ricker(100, 4)

plt.title('Ricker wavelet')
plt.plot(list(range(-50, 50)), vec)
plt.show()

import numpy as np
xx = np.linspace(-6, 6, 200)
y1 = np.sin(xx)
y2 = np.sin(xx**2)
y3 = y1+y2
widths = np.arange(1, 51)

cwt = sig.cwt(y3, ricker, widths)
plt.plot(xx, y3)
plt.show()

im = plt.imshow(cwt, extent=[-1, 1, 1, widths[-1]], aspect='auto', cmap='viridis')
plt.colorbar(im)
plt.show()