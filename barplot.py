from matplotlib import pyplot as plt
import numpy as np

weight = [5, 16, 49, 50, 55, 56, 62, 63, 81, 117, 134,
          186, 191, 201, 212, 227, 229];
scaled = [0.0118, 0.0209, 0.0225, 0.0357, 0.0379, 0.0511, 0.0571, 0.0578, 0.0603, 0.0625, 0.0673,
          0.0742, 0.0779, 0.0854, 0.0889, 0.0920, 0.0970];
index = np.arange(len(weight))

fig = plt.figure(1)
plt.bar(index, scaled)
plt.xticks(index, weight)
plt.xlabel("Weight")
plt.ylabel("Contribution Ratio")
plt.title("Weight - Contribution Relationship")
plt.show()

weight = [5, 16, 49, 50, 55, 56, 62, 63, 81, 117, 134,
          186, 191, 201, 212, 227, 229]
scaled = [0.0091, 0.0141, 0.0264, 0.0289, 0.0298, 0.0301, 0.0319, 0.0335, 0.0405, 0.0558,
          0.0708, 0.0942, 0.0962, 0.1026, 0.1088, 0.1108, 0.1135]
non_scaled = [0.0511, 0.0526, 0.0544, 0.0543, 0.0547, 0.0541, 0.0543, 0.0554, 0.0555, 0.0591,
              0.0608, 0.0647, 0.0639, 0.0649, 0.0661, 0.0675, 0.0666]
fig.savefig("bar1.png")

fig = plt.figure(2)
plt.subplot(1, 2, 1)
plt.bar(index, non_scaled)
plt.text(1, .1, "R2 Score : 0.9916")
plt.xticks(index, weight, rotation = 90)
plt.ylim([0, 0.15])
plt.xlabel("Weight")
plt.ylabel("Contribution Ratio")
plt.title("Non-Scaled")

plt.subplot(1, 2, 2)
plt.bar(index, scaled)
plt.text(1, .1, "R2 Score : 0.9977")
plt.xticks(index, weight, rotation = 90)
plt.ylim([0, 0.15])
plt.xlabel("Weight")
plt.ylabel("Contribution Ratio")
plt.title("MinMax-Scaled")

plt.show()
fig.savefig("barplot.png")