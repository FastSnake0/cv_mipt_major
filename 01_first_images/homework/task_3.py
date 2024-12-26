import cv2
import numpy as np


import cv2
import numpy as np

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def rotate(image, point: tuple, angle: float) -> np.ndarray:
    """
    Повернуть изображение по часовой стрелке на угол от 0 до 360 градусов и преобразовать размер изображения.

    :param image: исходное изображение
    :param point: значение точки (x, y), вокруг которой повернуть изображение
    :param angle: угол поворота
    :return: повернутное изображение
    """
    h, w = image.shape[:2]
    img_c = (w / 2, h / 2)

    rot = cv2.getRotationMatrix2D(img_c, angle, 1)

    rad = np.radians(angle)
    sin = np.sin(rad)
    cos = np.cos(rad)
    b_w = int((h * abs(sin)) + (w * abs(cos)))
    b_h = int((h * abs(cos)) + (w * abs(sin)))

    rot[0, 2] += ((b_w / 2) - img_c[0])
    rot[1, 2] += ((b_h / 2) - img_c[1])

    outImg = cv2.warpAffine(image, rot, (b_w, b_h), flags=cv2.INTER_LINEAR)
    return outImg

def apply_warpAffine(image, points1, points2) -> np.ndarray:
    """
    Применить афинное преобразование согласно переходу точек points1 -> points2 и
    преобразовать размер изображения.

    :param image: исходное изображение
    :param points1: исходные координаты точек
    :param points2: конечные координаты точек
    :return: преобразованное изображение
    """
    M = cv2.getAffineTransform(points1, points2)
    h, w, _ = image.shape
    x1y1 = M @ np.array([0,0,1]).T
    x1y2 = M @ np.array([0, h, 1]).T
    x2y1 = M @ np.array([w, 0, 1]).T
    x2y2 = M @ np.array([w, h, 1]).T

    border_points = np.vstack((x1y1, x1y2, x2y1, x2y2))
    x_min = min(border_points[:,0])
    x_max = max(border_points[:,0])
    y_min = min(border_points[:,1])
    y_max = max(border_points[:,1])

    M[0, 2] -= x_min
    M[1, 2] -= y_min

    image = cv2.warpAffine(image.copy(), M, (round(x_max - x_min), round(y_max - y_min)))

    return image
