import numpy as np
import matplotlib.pyplot as plt

heart = np.array([[0.1, 0.5], [0.1, 0.65], [0.3, 0.85], [0.5, 0.65], [0.7, 0.85], [0.9, 0.65], [0.9, 0.5], [0.5, 0], [0.1, 0.5]])
plus = np.array([[0.3, 0.5], [0.7, 0.5], [0.7, 0.7], [0.5, 0.7], [0.5, 1], [0.3, 1], [0.3, 0.7], [0.1, 0.7], [0.1, 0.5], [0.3, 0.5]])

def rotation_vector(vector, angle):
    angle = np.radians(angle)
    cos = np.cos(angle)
    sin = np.sin(angle)
    rotation_matrix = np.array([[cos, -sin], [sin, cos]])
    return np.dot(vector, rotation_matrix)

def scale(object, const):
    scaled_object = object * const
    return scaled_object

def mirror_by_axis(object, axis):
    mirrored_object = np.copy(object)
    if axis == 'x':
        mirrored_object[:, 1] *= -1 
    elif axis == 'y':
        mirrored_object[:, 0] *= -1  
    return mirrored_object

def tilt_figure(figure, axis, factor):
    if axis == 'x':
        new_matrix = np.array([[1, factor], [0, 1]])
    elif axis == 'y':
         new_matrix= np.array([[1, 0], [factor, 1]])
    transformed_figure = np.dot(figure,new_matrix )
    return transformed_figure

def transform(figure, transform_matrix):
    transformed_figure = np.dot(figure, transform_matrix)
    return transformed_figure


#1 функція, що обертає вектор на певний кут.
rotated_figure= rotation_vector(heart, 90)
plt.plot(rotated_figure[:, 0], rotated_figure[:, 1], 'r.-')
plt.title('Rotation')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()
print(rotated_figure)
print()

rotated_figure= rotation_vector(plus, 90)
plt.plot(rotated_figure[:, 0], rotated_figure[:, 1], 'b.-')
plt.title('Rotation')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()
print(rotated_figure)
print()

#2 функція масштабування фігури
scaled_figure = scale(heart, 2)

plt.plot(heart[:, 0], heart[:, 1], 'b.-', label='Original')
plt.plot(scaled_figure[:, 0], scaled_figure[:, 1], 'r.-', label='Scaled')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()
print(scaled_figure)
print()

scaled_figure = scale(plus, 2)
plt.plot(plus[:, 0], plus[:, 1], 'b.-', label='Original')
plt.plot(scaled_figure[:, 0], scaled_figure[:, 1], 'r.-', label='Scaled')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()
print(scaled_figure)
print()

#3 функція для відзеркалення відносно осі
mirrored_figure= mirror_by_axis(heart, 'x')
plt.plot(mirrored_figure[:, 0],mirrored_figure[:, 1], 'r.-')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()
print(mirrored_figure)
print()

mirrored_figure = mirror_by_axis(plus, 'y')
plt.plot(mirrored_figure[:, 0], mirrored_figure[:, 1], 'b.-')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()
print(mirrored_figure)
print()

# 4 функція нахилення осі координат
transformed_figure = tilt_figure(heart, 'y', -0.5)
plt.plot(transformed_figure[:, 0], transformed_figure[:, 1], 'r-')
plt.grid(True)
plt.title('Tilt axis')
plt.show()
print(transformed_figure)
print()

transformed_figure = tilt_figure(plus, 'x', -0.5)
plt.plot(transformed_figure[:, 0], transformed_figure[:, 1], 'b-')
plt.grid(True)
plt.title('Tilt axis')
plt.show()
print(transformed_figure)
print()

#5 функція для загальної трансформації через матрицю
custom_matrix = np.array([[5, 0],
                          [0, 6]])
result_figure = transform(heart, custom_matrix)

plt.plot(result_figure[:, 0], result_figure[:, 1], 'r-')
plt.title('Transformed figure')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()
print(result_figure)
print()

result_figure = transform(plus, custom_matrix)
plt.plot(result_figure[:, 0], result_figure[:, 1], 'b-')
plt.title('Transformed figure')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()
print(result_figure)
print()

# Трьохвимірна фігура - куб, та 2 його трансформації
vertices = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1]
])
edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7)
]
scale_matrix = np.array([
    [0.5, 0, 0],
    [0, 0.5, 0],
    [0, 0, 0.5]
])

rotation_matrix = np.array([
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1]
])
fig = plt.figure()
scaled_figure = transform(vertices, scale_matrix)
ax = fig.add_subplot(projection='3d')
ax.set_title('After scaling')
for edge in edges:
    start, end = edge
    x = [scaled_figure[start][0], scaled_figure[end][0]]
    y = [scaled_figure[start][1], scaled_figure[end][1]]
    z = [scaled_figure[start][2], scaled_figure[end][2]]
    ax.plot(x, y, z, color='r')
plt.show()
print(scaled_figure)
print()

fig1 = plt.figure()
rotation_figure = transform(vertices, rotation_matrix)
ax1 = fig1.add_subplot(projection='3d')
ax1.set_title('After rotation')
for edge in edges:
    start, end = edge
    x = [rotation_figure[start][0], rotation_figure[end][0]]
    y = [rotation_figure[start][1], rotation_figure[end][1]]
    z = [rotation_figure[start][2], rotation_figure[end][2]]
    ax1.plot(x, y, z, color='g')
plt.show()
print(rotation_figure)
