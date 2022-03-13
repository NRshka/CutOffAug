from typing import Union, Tuple
import numpy as np
from dataclasses import dataclass
from matplotlib import pyplot as plt
import cv2
import copy


@dataclass
class Point:
    x: Union[float, int]
    y: Union[float, int]

    def to_type(self, dst_type):
        self.x = dst_type(self.x)
        self.y = dst_type(self.y)

        return self

    def tuple(self):
        return self.x, self.y


def find_line_equation(p1: Point, p2: Point) -> Tuple[float, float]:
    if p1.x > p2.x:
        p1, p2 = p2, p1

    # line equation is y = k * x + b
    k = (p2.y - p1.y) / (p2.x - p1.x)
    b = p2.y - k * p2.x

    return k, b


def bound_edge_point(init_x: float, height: int, func_y_of_x, func_x_of_y) -> Point:
    est_y = func_y_of_x(init_x)

    est_y = max(est_y, 0)
    est_y = min(est_y, height - 1)

    est_x = func_x_of_y(est_y)

    return Point(est_x, est_y)


def find_line_points(func_y_of_x, func_x_of_y, width: int, height: int):
    left_point = bound_edge_point(0, height, func_y_of_x, func_x_of_y)
    right_point = bound_edge_point(width - 1, height, func_y_of_x, func_x_of_y)

    return left_point, right_point


def define_functions(k, b):
    func_y_of_x = lambda x: k * x + b
    func_x_of_y = lambda y: (y - b) / k

    return func_y_of_x, func_x_of_y


def draw_rectangle(p1: Point, p2: Point):
    plt.plot([p1.x, p2.x], [p1.y, p1.y])
    plt.plot([p2.x, p2.x], [p1.y, p2.y])
    plt.plot([p1.x, p2.x], [p2.y, p2.y])
    plt.plot([p1.x, p1.x], [p1.y, p2.y])


def find_polygon_points(
    func_y_of_x, width: int, height: int, p1: Point, p2: Point
):
    # if p1.x = 0
    # TODO здесь мы рисуем matplotlib, у него начало координа внизу
    # в opencv начало координат вврху
    # поэтому порядок в vehicles нужно будет изменить
    # import pdb
    # pdb.set_trace()
    vehicles = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ])

    if p1.y == 0:
        vehicles = np.array([
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
            [0, 0],
        ])
    elif p1.y == height - 1:
        vehicles = np.array([
            [0, height - 1],
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
        ])

    points_upon = [np.array([p1.x, p1.y])]
    points_below = [np.array([p2.x, p2.y])]

    for point in vehicles:
        line_y = func_y_of_x(point[0])

        if point[1] < line_y:
            points_upon.append(point)
        else:
            points_below.append(point)

    points_upon.append(np.array([p2.x, p2.y]))
    points_below.append(np.array([p1.x, p1.y]))

    return np.array(points_upon), np.array(points_below)


def generate_mask(width, height, size):
    mask = np.zeros((height, width))
    cls_idx = 0
    for x in range(0, width, 2 * size):
        for y in range(0, height, 2 * size):
            mask[y: y + size, x: x + size] = cls_idx
            cls_idx += 1

    return mask


def get_polygon_mask(points, width, height):
    points = np.array(points).astype(np.int32)

    mask = np.zeros((height, width))
    mask = cv2.fillPoly(mask, [points], 1)

    return mask == 1


def cut_mask_off(mask, polygon_mask):
    mask = copy.deepcopy(mask)
    indexes = np.unique(mask)[1:]
    for cls_index in indexes:
        class_mask = mask == cls_index
        intersection = class_mask & polygon_mask

        if intersection.mean() / class_mask.mean() > 0.1:
            mask[class_mask] = 0

    return mask


def calculate_polygon_area(points: np.ndarray):
    x = points[:, 0]
    y = points[:, 1]

    col1 = (x[:-1] * y[1:]).sum()
    col2 = (y[:-1] * x[1:]).sum()

    return 0.5 * col1 * col2


width = 256
height = 512
image = np.random.randn(height, width, 3)
mask = generate_mask(width, height, 20)


k, b = find_line_equation(
    Point(np.random.randint(0, width), np.random.randint(0, height)),
    Point(np.random.randint(0, width), np.random.randint(0, height)),
)
func_y_of_x, func_x_of_y = define_functions(k, b)
left_point, right_point = find_line_points(
    func_y_of_x, func_x_of_y, width, height
)

points_upon, points_below = find_polygon_points(
    func_y_of_x, width, height, left_point, right_point
)

cut_mask_smaller = get_polygon_mask(points_upon, width, height)
cut_mask_bigger = get_polygon_mask(points_below, width, height)

if calculate_polygon_area(cut_mask_smaller) > calculate_polygon_area(cut_mask_bigger):
    cut_mask_smaller, cut_mask_bigger = cut_mask_bigger, cut_mask_smaller

cutted_mask_small = cut_mask_off(mask, cut_mask_smaller)
cutted_mask_big = cut_mask_off(mask, cut_mask_bigger)

(
    fig,
    (
        cutmask_small_ax,
        mask_off_small_ax,
        cutmask_big_ax,
        mask_off_big_ax
    )
 ) = plt.subplots(1, 4)

cutmask_small_ax.imshow(cut_mask_smaller)
mask_off_small_ax.imshow(cutted_mask_small)
cutmask_big_ax.imshow(cut_mask_bigger)
mask_off_big_ax.imshow(cutted_mask_big)
plt.show()
