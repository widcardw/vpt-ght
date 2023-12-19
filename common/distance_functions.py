import numpy as np
import editdistance

def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    '''欧氏距离'''
    return np.linalg.norm(x - y)

'''编辑距离，需要使用 `pip install editdistance` 安装'''
edit_distance = editdistance.eval
