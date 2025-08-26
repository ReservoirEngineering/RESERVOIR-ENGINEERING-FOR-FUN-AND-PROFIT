import matplotlib.pyplot as plt
import numpy as np
N = 3000
d1 = np.random.randint(1, 7, N)
d2 = np.random.randint(1, 7, N)
S = d1 + d2

# Theoretical exact probabilities for sum of two fair dice
sums = np.arange(2, 13)  # 2 through 12 -> 11 values
counts = np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1])
theoretical_probs = counts / 36

# Plot histogram first
plt.hist(S, bins=np.arange(2, 14) - 0.5, 
         edgecolor='black', 
         density=True, 
         color='green', 
         label='Simulation')

# Overlay red asterisks
plt.plot(sums, theoretical_probs, 'r*', markersize=16, label='Exact probabilities')

# Labels and title with font size
plt.xlabel('Sum of two dice', fontsize=14)
plt.ylabel('Probability', fontsize=14)
plt.title('Histogram of Dice Sums with Exact Values', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.show()
