import numpy as np
h = 9
w = 10
strength = 0.2
corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
print(corners)
distortion = int(min(h, w) * strength)
print(distortion)
print(9*0.2)
print(int(9*0.2))