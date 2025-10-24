# Autocorrelation using numpy.correlate

import numpy as np
import matplotlib.pyplot as plt

# Generate a sample time series
data = np.sin(np.linspace(0, 4 * np.pi, 100)) + np.random.randn(100) * 0.2

# Remove the mean for proper autocorrelation calculation
data_demeaned = data - np.mean(data)

# Calculate autocorrelation using numpy.correlate
# mode='full' returns the full discrete linear cross-correlation
autocorr = np.correlate(data_demeaned, data_demeaned, mode='full')

# Normalize by the zero-lag value to get the autocorrelation function (ACF)
# The zero-lag value is at the center of the 'full' output
autocorr /= autocorr[len(data) - 1]

# Generate lag values
lags = np.arange(-len(data) + 1, len(data))

plt.figure(figsize=(10, 5))
plt.stem(lags, autocorr)
plt.title("Autocorrelation Function (ACF) using numpy.correlate")
plt.xlabel("Lag")
plt.ylabel("Autocorrelation Coefficient")
plt.grid(True)
plt.show()