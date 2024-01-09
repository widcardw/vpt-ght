import numpy as np
# import editdistance

def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    '''欧氏距离'''
    return np.linalg.norm(x - y)

'''编辑距离，需要使用 `pip install editdistance` 安装'''
# 但是这个库认为 replace 的代价为 1，因此重写了一个
# edit_distance = editdistance.eval

'''如果进行了替换，则距离+2，插入或删除则距离+1'''
def my_edit_distance(x: str, y: str) -> float:
    x_len = len(x)
    y_len = len(y)
    dp = np.zeros((x_len + 1, y_len + 1), dtype=int)
    dp[0, :] = np.arange(y_len + 1)
    dp[:, 0] = np.arange(x_len + 1)

    for i in range(1, x_len + 1):
        for j in range(1, y_len + 1):
            expense = 0 if x[i - 1] == y[j - 1] else 2
            dp[i, j] = dp[i - 1, j - 1] + expense
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i, j])

    return dp[x_len, y_len]
