import numpy as np
import matplotlib.pyplot as plt

group_xy_pos_data = np.load('group_xy_pos_data.npy')
group_waypt_data = np.load('group_waypt_data.npy')

plt.figure(figsize=(5, 5))
vector_length = 1
iniyaw = 0
oript_data = np.zeros(2)
waypt_data = group_waypt_data[0]
x_pos_data = group_xy_pos_data[:, 0]
y_pos_data = group_xy_pos_data[:, 1]
vec_endpt = oript_data + vector_length * np.array([np.cos(iniyaw), np.sin(iniyaw)])
plt.plot([oript_data[0], vec_endpt[0]], [oript_data[1], vec_endpt[1]], 'r-', label='forward direction')
plt.scatter([oript_data[0]], [oript_data[1]], color='blue', label='original point')
plt.scatter([waypt_data[0]], [waypt_data[1]], color='green', label='waypoint')
plt.plot(x_pos_data, y_pos_data, marker='.', linestyle='None', color='black', label='position data')
plt.axis('equal')
plt.title('x-y position')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.show()