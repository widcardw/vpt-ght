import numpy as np
from typing import Iterable, Callable, Union, TypeVar
from common.types import _T

from common.distance_functions import euclidean_distance

class VPTInternalNode:
    def __init__(self, 
                 pivot: _T, 
                 split_radius: float, 
                 left: Union['VPTInternalNode', 'PivotTable'], 
                 right: Union['VPTInternalNode', 'PivotTable'],
                 depth = 0):
        '''
        树的节点
        
        left: 左子树
        right: 右子树
        splitRadius: 划分半径
        pivot: 支撑点
        '''
        self.left = left
        self.right = right
        self.split_radius = split_radius
        self.pivot = pivot
        self.depth = depth


class PivotTable:
    def __init__(self, data: Iterable[_T], depth = 0):
        '''
        支撑点表

        data: 顶点数组
        depth: 该节点在树中的深度
        '''
        self.data = data
        self.depth = depth

# TypeAlias 要 Python 3.12 才能支持 ¯\_(ツ)_/¯
# VPTreeNode = TypeAlias('VPTreeNode', Union[VPTInternalNode, PivotTable])
VPTreeNode = TypeVar('VPTreeNode', VPTInternalNode, PivotTable)

def _select_pivot(data: Iterable[_T]) -> _T:
    '''选择支撑点，随机选取'''
    return data[np.random.choice(len(data))]

def determine_radius(distances: np.ndarray) -> float:
    '''确定划分半径，使用距离的中位数'''
    return np.median(distances)
    
def _exclude_pivot(data, pivot: _T) -> _T:
    '''将 pivot 从 data 中删除'''
    # 因为 numpy 和原生 list 方法不是非常一致，所以做了一下拆分
    if isinstance(data, np.ndarray):
        return np.delete(data, np.where(np.all(data == pivot, axis=1)), axis=0)
    elif isinstance(data, Iterable):
        return list(filter(lambda x: x != pivot, data))
    else:
        raise NotImplementedError

def _build_vptree_recursive(data: Iterable[_T],
                            dist_fn: Callable[[_T, _T], float],
                            depth = 0,
                            max_leaf_size = 10
                            ) -> VPTreeNode:
    '''
    内部调用，递归建立 VP 树

    data: 顶点数组
    dist_fn: 距离函数
    depth: 该节点在树中的深度
    max_leaf_size: 支撑点表的最大支撑点数
    '''
    if len(data) <= max_leaf_size:
        return PivotTable(data, depth)
    vp = _select_pivot(data)
    data = _exclude_pivot(data, vp)
    # 计算所有点到 vp 的距离
    distances = np.array([dist_fn(d, vp) for d in data])
    radius = determine_radius(distances)
    # 根据距离划分数据，分为左右子树
    if isinstance(data, np.ndarray):
        left_data = np.empty((0, data.shape[1]))
        right_data = np.empty((0, data.shape[1]))
        for point, dist in zip(data, distances):
            point = np.expand_dims(point, axis=0)
            if dist <= radius:
                left_data = np.append(left_data, point, axis=0)
            else:
                right_data = np.append(right_data, point, axis=0)
    elif isinstance(data, Iterable):
        left_data = []
        right_data = []
        for point, dist in zip(data, distances):
            if dist <= radius:
                left_data.append(point)
            else:
                right_data.append(point)
    left = _build_vptree_recursive(left_data, dist_fn, depth + 1)
    right = _build_vptree_recursive(right_data, dist_fn, depth + 1)
    return VPTInternalNode(vp, radius, left, right, depth)

def build_vptree(points: Iterable[_T],
                 dist_fn: Callable[[_T, _T], float] = euclidean_distance,
                 max_leaf_size = 10
                 ) -> VPTreeNode:
    '''建立 VP 树'''
    return _build_vptree_recursive(points, dist_fn, max_leaf_size)

def _pivot_table_search(node: PivotTable, 
                        query: _T, 
                        radius: float, 
                        dist_fn: Callable[[_T, _T], float] = euclidean_distance):
    '''在支撑点表内搜索'''
    for point in node.data:
        if (dist := dist_fn(point, query)) <= radius:
            yield point, dist, node.depth

def _get_all_data(node: VPTreeNode,
                  query: _T,
                  dist_fn: Callable[[_T, _T], float] = euclidean_distance):
    if isinstance(node, PivotTable):
        for point in node.data:
            yield point, dist_fn(point, query), node.depth
    else:
        yield from _get_all_data(node.left, query, dist_fn)
        yield from _get_all_data(node.right, query, dist_fn)

def vp_range_query(node: VPTreeNode, 
                 query: _T, 
                 radius: float, 
                 dist_fn: Callable[[_T, _T], float] = euclidean_distance):
    # 在支撑点表内搜索
    if isinstance(node, PivotTable):
        yield from _pivot_table_search(node, query, radius, dist_fn)
    # 支撑点是查询结果
    else:
        if (dist := dist_fn(query, node.pivot)) <= radius:
            yield node.pivot, dist, node.depth
        # 球内数据全部是查询结果
        if dist_fn(query, node.pivot) + node.split_radius <= radius:
            yield from _get_all_data(node.left, query, dist_fn)
        # 球内侧不能排除
        elif dist_fn(query, node.pivot) - node.split_radius <= radius:
            yield from vp_range_query(node.left, query, radius, dist_fn)
        # 球外侧不能排除
        if dist_fn(node.pivot, query) + radius > node.split_radius:
            yield from vp_range_query(node.right, query, radius, dist_fn)
