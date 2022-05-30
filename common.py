# author: chenjianwei
## Write only functions, not classes if necessary
# my: 2731289611338
# def ej():
#     import pathlib
#     exec(pathlib.Path("/Users/didi/Desktop/repos/dijk_nn/common.py").open('r').read(), globals())

import collections
import functools
import itertools
import pathlib
import random
import time

import dill
import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import redis
import torch
import torch.nn as nn
from scipy.spatial import KDTree

STATIC_LOCATION_BUAA                    = [116.34762, 39.97692]
STATIC_LOCATION_SHANGHAI_SUZHOU_MIDDLE  = [120.95164, 31.28841]
STATIC_LOCATION_SHENYANG                = [123.38, 41.8]

DATA_PATH = "/Users/didi/Desktop/data"

def data_path_(p):
    return str((pathlib.Path(DATA_PATH) / p).absolute())

def iterdir(obj):
    df = pd.DataFrame(dir(obj), columns=['member'])
    df = df.query("not member.str.startswith('_')")
    lines = df \
        .assign(first_letter=df.member.apply(lambda x: str(x)[0])) \
        .groupby('first_letter') \
        .member.unique() \
        .apply(lambda xs: ", ".join(xs)) \
        .to_numpy() \
        .tolist()
    for line in lines:
        yield line

def printdir(obj):
    for line in iterdir(obj):
        print("-", line)

def head(jter, n=10):
    more = False
    for i, line in enumerate(jter):
        print(line)
        if i >= n:
            more = True
            break
    if more:
        print("... for head")

def take(n):
    def func(xs):
        if n >= xs.__len__():
            return None
        return xs[n]
    return func
for i in range(10):
    globals()[f'take_{i}'] = take(i)

def monkey(_class, method_name=None):
    def _decofunc(func):
        if not method_name:
            _method_name = func.__name__
            if _method_name.startswith('_'):
                _method_name = _method_name[1:]
        else:
            _method_name = method_name
        setattr(_class, _method_name, func)
        return func
    return _decofunc


class PandasIndexContext(object):
    def __init__(self, df, index_key):
        self.df = df
        self.index_key = index_key
        self.old_index_key = None
    def __enter__(self):
        df = self.df
        old_index_key = self.df.index.name
        if not old_index_key:
            old_index_key = "index"
        self.old_index_key = old_index_key
        df = df \
            .reset_index() \
            .set_index(self.index_key)
        self.df = df
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        df = self.df
        df = df \
            .reset_index() \
            .set_index(self.old_index_key)
        self.df = df

@monkey(pd.DataFrame, "index_context")
def _pandas_index_context(df, index_key):
    return PandasIndexContext(df, index_key)


@monkey(pd.DataFrame, "assign_by")
def _pandas_assign_by(df, key, **kwargv)->pd.DataFrame:
    with df.index_context(key) as ctx:
        df = ctx.df
        df = df.assign(**kwargv)
        ctx.df = df
    return ctx.df

def _np_distance_args_preprocess(*args):
    assert len(args) in (1, 4)
    if args.__len__() == 4:
        lng1, lat1, lng2, lat2 = args
    if args.__len__() == 1:
        gps_arr = args[0]
        lng1, lat1, lng2, lat2 = [gps_arr[:, i] for i in range(4)]
    return lng1, lat1, lng2, lat2

def np_distance(*args):
    lng1, lat1, lng2, lat2 = _np_distance_args_preprocess(*args)
    radius = 6371
    dlng = np.radians(lng2-lng1)
    dlat = np.radians(lat2-lat1)
    a = np.sin(dlat/2)*np.sin(dlat/2) \
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlng/2) * np.sin(dlng/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = radius * c
    return d
def np_coords(*args):
    lng1, lat1, lng2, lat2 = _np_distance_args_preprocess(*args)
    sx = np.copysign(1, lng1-lng2)
    sy = np.copysign(1, lat1-lat2)
    x = np_distance(lng1, lat1, lng2, lat1)
    y = np_distance(lng1, lat1, lng1, lat2)
    return np.stack([sx*x, sy*y], axis=0).T
def np_mht(*args):
    r = np_coords(*args)
    y = np.abs(r[:, 0]) + np.abs(r[:, 1])
    return y
def torch_mht(x):
    return torch.tensor(np_mht(x.numpy()))

def tensor_to_shape_with_zero(x, shape):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    x_zero = torch.zeros(shape)
    x = x[:shape[0], :shape[1]]
    x = torch.cat([
        x,
        x_zero[..., :x.shape[1]]
    ], axis=0)
    x = x[:shape[0], :shape[1]]
    x = torch.cat([
        x,
        x_zero[:x.shape[0], ...]
    ], axis=1)
    x = x[:shape[0], :shape[1]]
    return x


@monkey(torch.Tensor)
def remove_zero(x):
    first_column = x[..., 0].tolist()
    first_row = x[0, ...].tolist()
    a1 = x.shape[0] if 0 not in first_column else first_column.index(0)
    a2 = x.shape[1] if 0 not in first_row else first_row.index(0)
    return x[:a1, :a2]


if __name__ == '__main__':
    print(torch_mht(torch.rand(10,4)))
    tensor = torch.tensor([1,2,3,4,0,0,0,0]).view(-1, 2)
    print(tensor)
    print(tensor_to_shape_with_zero(tensor, (10, 2)))
    print(tensor.remove_zero())