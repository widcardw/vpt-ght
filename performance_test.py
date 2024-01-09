import numpy as np
from vptree.vptree import build_vptree, vp_range_query
from ghtree.ghtree import build_ghtree, gh_range_query
import string
import random
from typing import Iterable
from common.distance_functions import euclidean_distance, my_edit_distance as edit_distance
import time

def test_vptree_vector(points: np.ndarray, queries: np.ndarray, radius: float):
    t = time.time()
    vpt = build_vptree(points)
    end_t = time.time()
    print(f'VP Tree with {points.shape} vectors built in {end_t - t}s')

    t = time.time()
    for query in queries:
        result_set = []
        results_it = vp_range_query(vpt, query, radius)
        for result in results_it:
            result_set.append(result)
    end_t = time.time()
    print(f'{len(queries)} queries finished in {end_t - t}s')

def test_ghtree_vector(points: np.ndarray, queries: np.ndarray, radius: float):
    t = time.time()
    vpt = build_ghtree(points)
    end_t = time.time()
    print(f'GH Tree with {points.shape} vectors built in {end_t - t}s')

    t = time.time()
    for query in queries:
        result_set = []
        results_it = gh_range_query(vpt, query, radius)
        for result in results_it:
            result_set.append(result)
    end_t = time.time()
    print(f'{len(queries)} queries finished in {end_t - t}s')

def test_vector():
    point_num, dim = 10_000, 10
    query_num = 500
    radius = 1.
    points = np.random.rand(point_num, dim)
    queries = np.random.rand(query_num, dim)
    test_ghtree_vector(points, queries, radius)
    test_vptree_vector(points, queries, radius)

def _generate_rand_string(length = 10):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

def test_vptree_edit(points: Iterable[str], queries: Iterable[str], radius: float):
    t = time.time()
    vpt = build_vptree(points, edit_distance)
    end_t = time.time()
    print(f'VP Tree with {len(points)} strings built in {end_t - t}s')

    t = time.time()
    for query in queries:
        result_set = []
        results_it = vp_range_query(vpt, query, radius, edit_distance)
        for result in results_it:
            result_set.append(result)
    end_t = time.time()
    print(f'{len(queries)} queries finished in {end_t - t}s')


def test_ghtree_edit(points: Iterable[str], queries: Iterable[str], radius: float):
    t = time.time()
    ght = build_ghtree(points, edit_distance)
    end_t = time.time()
    print(f'GH Tree with {len(points)} strings built in {end_t - t}s')
    for query in queries:
        result_set = []
        results_it = gh_range_query(ght, query, radius, edit_distance)
        for result in results_it:
            result_set.append(result)
    end_t = time.time()
    print(f'{len(queries)} queries finished in {end_t - t}s')

def test_edit():
    point_num = 2_000
    query_num = 100
    radius = 15
    min_length = 10
    max_length = 20
    points = [_generate_rand_string(random.randrange(min_length, max_length))
              for _ in range(point_num)]
    queries = [_generate_rand_string(random.randrange(min_length, max_length))
               for _ in range(query_num)]
    
    test_ghtree_edit(points, queries, radius)
    test_vptree_edit(points, queries, radius)
    

if __name__ == '__main__':
    test_vector()
    test_edit()

# python ./performance_test.py >> performance.log
