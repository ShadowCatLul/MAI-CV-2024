import cv2
import numpy as np


def rotate(image, point: tuple, angle: float) -> np.ndarray:
    h, w, _ = image.shape

    M = cv2.getRotationMatrix2D(point, angle, scale=1.0)
    
    points = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]])

    transformed_corners = M @ points.T
    min_x = np.min(transformed_corners[0, :])
    max_x = np.max(transformed_corners[0, :])
    min_y = np.min(transformed_corners[1, :])
    max_y = np.max(transformed_corners[1, :])

    new_w = int(np.round(max_x - min_x))
    new_h = int(np.round(max_y - min_y))


    least_coords = [min_x, min_y]
    resized= [new_w, new_h]

    M[0, 2] -= least_coords[0]
    M[1, 2] -= least_coords[1]

    rot_img = cv2.warpAffine(image, M, resized)
    return rot_img


def apply_warpAffine(image, points1, points2) -> np.ndarray:
    """
    Применить афинное преобразование согласно переходу точек points1 -> points2 и
    преобразовать размер изображения.

    :param image:
    :param points1:
    :param points2:
    :return: преобразованное изображение
    """
    M = cv2.getPerspectiveTransform(np.float32(points1), np.float32(points2))
    
    d_width = int(max(points2[0][0], points2[1][0], points2[2][0], points2[3][0]))
    d_height = int(max(points2[0][1], points2[1][1], points2[2][1], points2[3][1]))
    image = cv2.warpPerspective(image, M, (d_width, d_height))


    return image
