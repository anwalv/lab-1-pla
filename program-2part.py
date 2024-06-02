import cv2
import numpy as np
import matplotlib.pyplot as plt
image1 = cv2.imread('photo_2024-06-02_01-45-04.jpg')
image2 = cv2.imread('photo_2024-06-02_01-45-08.jpg')

resized_image = cv2.resize(image2, (400, 700))
cv2.imshow('Resized Image',resized_image)
cv2.waitKey(0)
rotated_image = cv2.rotate(image1, cv2.ROTATE_90_CLOCKWISE)
cv2.imshow('Rotated Image',rotated_image)
cv2.waitKey(0)

heart = np.array([[0.1, 0.5], [0.1, 0.65], [0.3, 0.85], [0.5, 0.65], [0.7, 0.85], [0.9, 0.65], [0.9, 0.5], [0.5, 0], [0.1, 0.5]])
plus = np.array([[0.3, 0.5], [0.7, 0.5], [0.7, 0.7], [0.5, 0.7], [0.5, 1], [0.3, 1], [0.3, 0.7], [0.1, 0.7], [0.1, 0.5], [0.3, 0.5]])

def rotate_array(array, angle):
    rotation_matrix = cv2.getRotationMatrix2D((0, 0), angle, 1)
    result = cv2.transform(array.reshape(1, -1, 2), rotation_matrix)
    rotated_array = result.reshape(array.shape)

    # Візуалізація результату
    plt.figure()
    plt.plot(rotated_array[:, 0], rotated_array[:, 1], 'ro-', label='Rotated')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Rotation')
    plt.legend()
    plt.grid(True)
    plt.show()
    print(rotation_matrix)
    print()

rotate_array(plus, 90)
rotate_array(heart,90)
def scale_array(array, scale_factor):
    scale_matrix = np.array([[scale_factor, 0],
                             [0, scale_factor]], dtype=np.float32)
    scaled_figure = cv2.transform(array.reshape(1, -1, 2), scale_matrix).reshape(-1, 2)
    return scaled_figure
scaled_figure = scale_array(heart, 2)
plt.plot(heart[:, 0], heart[:, 1], 'bo-', label='Original')
plt.plot(scaled_figure[:, 0], scaled_figure[:, 1], 'go-', label='Scaled')
plt.xlabel('X')
plt.ylabel('Y')
plt.title("Scaled figure")
plt.legend()
plt.grid(True)
plt.show()
print(scaled_figure)
print()

scaled_figure = scale_array(plus, 2)
plt.plot(plus[:, 0], plus[:, 1], 'bo-', label='Original')
plt.plot(scaled_figure[:, 0], scaled_figure[:, 1], 'ro-', label='Scaled')
plt.xlabel('X')
plt.ylabel('Y')
plt.title("Scaled figure")
plt.legend()
plt.grid(True)
plt.show()
print(scaled_figure)
print()

def mirror_array(array, axis):
    if axis == 'x':
        mirror_matrix = np.array([[1, 0],
                                  [0, -1]], dtype=np.float32)
        title = 'Mirrored Array (X-axis)'
    elif axis == 'y':
        mirror_matrix = np.array([[-1, 0],
                                  [0, 1]], dtype=np.float32)
        title = 'Mirrored Array (Y-axis)'
    mirrored_array = cv2.transform(array.reshape(1, -1, 2), mirror_matrix).reshape(-1, 2)
    print(mirrored_array)

    plt.figure()
    plt.plot(array[:,0], array[:,1], 'bo-', label='Original')
    plt.plot(mirrored_array[:,0], mirrored_array[:,1], 'ro-', label=title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

mirror_array(plus, 'x')
mirror_array(heart, 'y')


def tilt_figure(figure, axis, factor):
    if axis == 'x':
        new_matrix = np.array([[1, factor], [0, 1]], dtype=np.float32)
    elif axis == 'y':
        new_matrix = np.array([[1, 0], [factor, 1]], dtype=np.float32)
    transformed_figure = cv2.transform(figure.reshape(1, -1, 2), new_matrix).reshape(-1, 2)
    return transformed_figure

transformed_figure = tilt_figure(plus, 'y', -0.5)
plt.plot(transformed_figure[:, 0], transformed_figure[:, 1], 'go-', label='Scaled')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Tilt axis y')
plt.legend()
plt.grid(True)
plt.show()
print(scaled_figure)
print()

transformed_figure = tilt_figure(heart, 'x', -0.5)
plt.plot(transformed_figure[:, 0], transformed_figure[:, 1], 'go-', label='Scaled')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Tilt axis x')
plt.legend()
plt.grid(True)
plt.show()
print(scaled_figure)
print()

def transform(figure, transform_matrix):
    result = cv2.gemm(figure.astype(np.float32), transform_matrix.astype(np.float32), 1, None, 0)
    return result

custom_matrix = np.array([[5, 0],
                          [0, 6]])
result_figure = transform(heart, custom_matrix)

plt.plot(result_figure[:, 0], result_figure[:, 1], 'ro-')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Transformed figure')
plt.grid(True)
plt.show()
print(result_figure)
print()

result_figure = transform(plus, custom_matrix)
plt.plot(result_figure[:, 0], result_figure[:, 1], 'go-')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Transformed figure')
plt.grid(True)
plt.show()
print(result_figure)
print()