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

rotated_figure= rotation_vector(plus, 90)
plt.plot(rotated_figure[:, 0], rotated_figure[:, 1], 'b.-')
plt.title('Rotation')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()

#2 функція масштабування фігури
scaled_figure = scale(heart, 2)

plt.plot(heart[:, 0], heart[:, 1], 'b.-', label='Original')
plt.plot(scaled_figure[:, 0], scaled_figure[:, 1], 'r.-', label='Scaled')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()

scaled_figure = scale(plus, 2)
plt.plot(plus[:, 0], plus[:, 1], 'b.-', label='Original')
plt.plot(scaled_figure[:, 0], scaled_figure[:, 1], 'r.-', label='Scaled')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()

#3 функція для відзеркалення відносно осі
mirrored_figure= mirror_by_axis(heart, 'x')
plt.plot(mirrored_figure[:, 0],mirrored_figure[:, 1], 'r.-')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()

mirrored_figure = mirror_by_axis(plus, 'y')
plt.plot(mirrored_figure[:, 0], mirrored_figure[:, 1], 'b.-')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()
# 4 функція нахилення осі координат
transformed_figure = tilt_figure(heart, 'y', -0.5)
plt.plot(transformed_figure[:, 0], transformed_figure[:, 1], 'r-')
plt.grid(True)
plt.title('Tilt axis')
plt.show()

transformed_figure = tilt_figure(plus, 'x', -0.5)
plt.plot(transformed_figure[:, 0], transformed_figure[:, 1], 'b-')
plt.grid(True)
plt.title('Tilt axis')
plt.show()


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

result_figure = transform(plus, custom_matrix)

plt.plot(result_figure[:, 0], result_figure[:, 1], 'b-')
plt.title('Transformed figure')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()

