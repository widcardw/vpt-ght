import numpy as np
from typing import Iterable, Callable, Union, Tuple, TypeVar
from common.types import _T
from common.distance_functions import euclidean_distance

'''
教材中支撑点表的数据结构是
class PivotTable {
    data[];
    pivot[];
    distance[][];
}
但是这里似乎用不着那么多数据结构？
如果说，保存了 distance 之后，后面就无须再计算，只需从表中查询即可
'''

class PivotTable:
    def __init__(self, data: Iterable[_T], depth = 0):
        '''
        支撑点表

        data: 顶点数组
        depth: 该节点在树中的深度
        '''
        self.data = data
        self.depth = depth

class GHTInternalNode:
    def __init__(self,
                 c1: _T,
                 c2: _T,
                 left: Union['GHTInternalNode', 'PivotTable'],
                 right: Union['GHTInternalNode', 'PivotTable'],
                 depth = 0
                 ):
        '''
        GH 树的内部节点
        c1: 支撑点 1
        c2: 支撑点 2
        left: 左子树
        right: 右子树
        depth: 该节点在树中的深度
        '''
        self.c1 = c1
        self.c2 = c2
        self.left = left
        self.right = right
        self.depth = depth

GHTreeNode = TypeVar('GHTreeNode', GHTInternalNode, PivotTable)

def build_ghtree(points: Iterable[_T],
                 dist_fn: Callable[[_T, _T], float] = euclidean_distance,
                 max_leaf_size = 10
                 ) -> GHTreeNode:
    '''
    建立 GH 树

    points: 顶点数组
    dist_fn: 距离函数
    max_leaf_size: 支撑点表的最大支撑点数
    '''
    return _build_ghtree_recursive(points, dist_fn, max_leaf_size)

def gh_range_query(node: GHTreeNode,
                query: _T,
                radius: float,
                dist_fn: Callable[[_T, _T], float] = euclidean_distance):
    '''
    在 GH 树中搜索
    
    node: GH 树
    query: 查询点
    radius: 查询半径
    dist_fn: 距离函数
    '''
    # 在支撑点表内搜索
    if isinstance(node, PivotTable):
        yield from _pivot_table_search(node, query, radius, dist_fn)
    else:
        # 支撑点 c1 在查询范围内（是查询结果）
        if (dist := dist_fn(query, node.c1)) <= radius:
            yield node.c1, dist, node.depth
        # 支撑点 c2 在查询范围内（是查询结果）
        if (dist := dist_fn(query, node.c2)) <= radius:
            yield node.c2, dist, node.depth
        # 无法排除左子树，则搜索左子树
        if dist_fn(query, node.c1) - dist_fn(query, node.c2) <= 2 * radius:
            yield from gh_range_query(node.left, query, radius, dist_fn)
        # 无法排除右子树，则搜索右子树
        if dist_fn(query, node.c2) - dist_fn(query, node.c1) < 2 * radius:
            yield from gh_range_query(node.right, query, radius, dist_fn)

def _build_ghtree_recursive(data: Iterable[_T],
                            dist_fn: Callable[[_T, _T], float],
                            depth = 0,
                            max_leaf_size = 10
                            ) -> GHTreeNode:
    '''
    内部调用，递归建立 GH 树

    data: 顶点数组
    dist_fn: 距离函数
    depth: 该节点在树中的深度
    max_leaf_size: 支撑点表的最大支撑点数
    '''
    # 若数据量小于最大支撑点数，则建立支撑点表
    if len(data) <= max_leaf_size:
        return PivotTable(data, depth)
    # 选择支撑点，此处采用的是随机选择方法
    c1, c2 = _select_pivot(data)
    # 将支撑点从数据中删除
    data = _exclude_pivot(data, c1, c2)
    # 根据距离划分数据，分为左右子树。由于 numpy 数组已经封装好了一些方法，所以进行了拆分
    if isinstance(data, np.ndarray):
        left_data = np.empty((0, data.shape[1]))
        right_data = np.empty((0, data.shape[1]))
        for point in data:
            point = np.expand_dims(point, axis=0)
            if dist_fn(point, c1) <= dist_fn(point, c2):
                left_data = np.append(left_data, point, axis=0)
            else:
                right_data = np.append(right_data, point, axis=0)
    elif isinstance(data, Iterable):
        left_data: list[_T] = []
        right_data: list[_T] = []
        for point in data:
            if dist_fn(point, c1) <= dist_fn(point, c2):
                left_data.append(point)
            else:
                right_data.append(point)
    # 递归建立左右子树
    left = _build_ghtree_recursive(left_data, dist_fn, depth + 1)
    right = _build_ghtree_recursive(right_data, dist_fn, depth + 1)
    return GHTInternalNode(c1, c2, left, right, depth)

def _select_pivot(data: Iterable[_T]) -> Tuple[_T, _T]:
    '''选择支撑点，随机选取'''
    random_indices = np.random.choice(len(data), 2, replace=False)
    return data[random_indices[0]], data[random_indices[1]]

def _exclude_pivot(data: Iterable[_T], c1: _T, c2: _T) -> Iterable[_T]:
    '''将 pivot 从 data 中删除'''
    if isinstance(data, np.ndarray):
        return np.delete(data, np.where(np.all(data == c1, axis=1) | np.all(data == c2, axis=1)), axis=0)
    elif isinstance(data, Iterable):
        return list(filter(lambda x: x != c1 and x != c2, data))
    else:
        raise NotImplementedError

def _pivot_table_search(node: PivotTable, 
                        query: _T, 
                        radius: float, 
                        dist_fn: Callable[[_T, _T], float] = euclidean_distance):
    '''在支撑点表内搜索'''
    for point in node.data:
        if (dist := dist_fn(point, query)) <= radius:
            yield point, dist, node.depth


