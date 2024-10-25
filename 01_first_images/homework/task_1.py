import cv2
import numpy as np
import pandas as pd


def find_way_from_maze(image: np.ndarray) -> tuple:

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    start = (0, np.where(binary[0] == 0)[0][0]) 
    end = (binary.shape[0] - 1, np.where(binary[-1] == 0)[0][0]) 


    from collections import deque

    queue = deque([start])
    visited = set()
    visited.add(start)
    parent = {start: None}

    while queue:
        current = queue.popleft()
        if current == end:
            break

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if (0 <= neighbor[0] < binary.shape[0] and
                0 <= neighbor[1] < binary.shape[1] and
                binary[neighbor] == 0 and 
                neighbor not in visited):
                visited.add(neighbor)
                queue.append(neighbor)
                parent[neighbor] = current

    path = []
    step = end
    while step is not None:
        path.append(step)
        step = parent[step]
    path.reverse()

    x, y = zip(*path)
    return (np.array(x), np.array(y))