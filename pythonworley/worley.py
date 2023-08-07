#!/usr/bin/env python
# -*- coding: utf8 -*-

import itertools
import numpy as np


def noisegrid(*shape, boundary=False, seed=None):
    """
    Generate grid noise. 
    Genretaes grod of noise in range (0,1) in each grid cell.
    
    Parameters
    ----------
    *shape
        Integers or tuple of integers
    boundary : bool, default False
        If True - pad with 1 layer of periodic bondary in each dimension
    seed : int, default None
        Numpy random seed

    Returns
    -------
    noise : ndarray
        Noise of shape = shape

    """
    np.random.seed(seed)
    shape = tuple(shape[0]) if isinstance(shape[0], (tuple, list)) else shape
    noise = np.random.uniform(0, 1, size=shape)
    if boundary:
        ndim = noise.ndim
        for i in range(ndim):
            n = noise.shape[i]
            slc = [slice(None)] * ndim
            slc[i] = slice(0,1)
            end = noise[tuple(slc)]
            slc = [slice(None)] * ndim
            slc[i] = slice(n-1,n)
            start = noise[tuple(slc)]
            noise = np.concatenate([start, noise, end], axis=i)
    return noise


def noisecoords(*shape, boundary=False, seed=None):
    """
    Utility function to generate grid noise coordinates. 
    Generates coordinate meshgrid and add noise in range (0,1).
    
    Parameters
    ----------
    *shape
        Integers or tuple of integers
    boundary : bool, default False
        If True - pad with 1 layer of periodic bondary in each dimension
    seed : int, default None
        Numpy random seed

    Returns
    -------
    coords : ndarray
        Coordinates of shape = (ndim, *shape), ndim = len(shape)

    """
    np.random.seed(seed)
    shape = tuple(shape[0]) if isinstance(shape[0], (tuple, list)) else shape
    xshape = tuple([s + 2 for s in shape]) if boundary else shape
    coords = []
    ndim = len(shape)
    for i in range(ndim):
        sh = np.ones((ndim)).astype(int)
        sh[i] = xshape[i]
        x = np.arange(xshape[i]).reshape(sh)
        x = x - 1 if boundary else x
        for j in range(ndim):
            if j != i:
                x = np.repeat(x, xshape[j], axis=j)
        x = x + noisegrid(shape, boundary=boundary, seed=seed)
        coords.append(x)
    coords = np.stack(coords)
    return coords


def gridcoords(*shape, dens=4):
    """
    Utility function to generate grid coordinates. 
    
    Parameters
    ----------
    *shape
        Integers or tuple of integers
    dens : int, default 4
        Number of points in each two shape nodes along any direction

    Returns
    -------
    grid : ndarray
        Grid coordinates of shape = (ndim, *shape * dens), ndim = len(shape)

    """
    shape = tuple(shape[0]) if isinstance(shape[0], (tuple, list)) else shape
    xshape = tuple([s * dens for s in shape])
    grid = []
    ndim = len(shape)
    for i in range(ndim):
        sh = np.ones((ndim)).astype(int)
        sh[i] = xshape[i]
        x = np.arange(xshape[i]).reshape(sh) / dens
        for j in range(ndim):
            if j != i:
                x = np.repeat(x, xshape[j], axis=j)
        grid.append(x)
    grid = np.stack(grid)
    return grid


def worley(*shape, dens=4, seed=None):
    """
    Utility function to generate Worley noise. 
    
    Parameters
    ----------
    *shape
        Integers or tuple of integers
    dens : int, default 4
        Number of points in each two shape nodes along any direction
    seed : int, default None
        Numpy random seed

    Returns
    -------
    dist : ndarray
        Distances of shape = (ndim, *shape * dens), ndim = len(shape)
        Sorted ascending along axis=0. The first is the smallest distance.
    cent : ndarray
        Corrdinates of grid noise centers of shape = (ndim, *shape)

    Example
    -------
    >>> from pythonworley import worley
    >>> shape = (32,32)
    >>> w, c = worley(shape, dens=8)
    >>> plt.imshow(w[0].T)
    >>> plt.scatter(*c)

    """    
    shape = tuple(shape[0]) if isinstance(shape[0], (tuple, list)) else shape
    ndim = len(shape)
    noise = noisecoords(shape, boundary=True, seed=seed)
    slc = [slice(None)] + [slice(1,-1)] * ndim
    cent = noise[tuple(slc)] * dens
    x = np.copy(noise)
    for i in range(1,ndim+1):
        noise = np.repeat(noise, dens, axis=i)
    grid = gridcoords(shape, dens=dens)
    dist = []
    ranges = [range(-1,2)] * ndim
    for direction in itertools.product(*ranges):
        sh = noise.shape
        slc = [slice(dens*(d+1),sh[i+1]+dens*(d-1)) for i, d in enumerate(direction)]
        slc = tuple([slice(None)] + slc)
        r = grid - noise[slc]
        r2 = np.sum(r**2, axis=0)
        dist.append(np.sqrt(r2))
    dist = np.stack(dist)
    dist = np.sort(dist, axis=0)
    return dist, cent





