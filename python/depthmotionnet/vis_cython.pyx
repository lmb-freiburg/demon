#
#  DeMoN - Depth Motion Network
#  Copyright (C) 2017  Benjamin Ummenhofer, Huizhong Zhou
#  
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport isfinite


@cython.boundscheck(False)
cdef _compute_point_cloud_from_depthmap(
    np.ndarray[np.float32_t, ndim=2] depth, 
    np.ndarray[np.float32_t, ndim=3] normals, 
    np.ndarray[np.uint8_t, ndim=3] colors, 
    np.ndarray[np.float32_t, ndim=2] K, 
    np.ndarray[np.float32_t, ndim=2] R_arr,
    np.ndarray[np.float32_t, ndim=1] t_arr ):

    cdef int valid_count
    valid_count = 0
    cdef int x, y
    cdef int index
    cdef int width, height
    cdef float d
    cdef float tmp[3]
    cdef float X[3]
    cdef float inv_fx = 1/K[0,0]
    cdef float inv_fy = 1/K[1,1]
    cdef float cx = K[0,2]
    cdef float cy = K[1,2]
    cdef float h = 0.5
    cdef float [:,:] R = R_arr
    cdef float [:] t = t_arr

    width = depth.shape[1]
    height = depth.shape[0]

    for y in range(height):
        for x in range(width):
            d = depth[y,x]
            if isfinite(d) and d > 0:
                valid_count += 1

    cdef np.ndarray[np.float32_t, ndim=2] points_arr = np.empty((valid_count,3), dtype=np.float32)
    cdef float [:,:] points = points_arr
    cdef np.ndarray[np.float32_t,ndim=2] normals_attr_arr = np.empty((valid_count,3), dtype=np.float32)
    cdef float [:,:] normals_attr = normals_attr_arr
    cdef np.ndarray[np.uint8_t,ndim=2] colors_attr_arr = np.empty((valid_count,3), dtype=np.uint8)
    cdef unsigned char [:,:] colors_attr = colors_attr_arr

    index = 0
    for y in range(height):
        for x in range(width):
            d = depth[y,x]
            if isfinite(d) and d > 0:
                tmp[0] = d*((x+h) - cx)*inv_fx - t[0]
                tmp[1] = d*((y+h) - cy)*inv_fy - t[1]
                tmp[2] = d - t[2]
                X[0] = R[0,0]*tmp[0] + R[1,0]*tmp[1] + R[2,0]*tmp[2]
                X[1] = R[0,1]*tmp[0] + R[1,1]*tmp[1] + R[2,1]*tmp[2]
                X[2] = R[0,2]*tmp[0] + R[1,2]*tmp[1] + R[2,2]*tmp[2]
                points[index,0] = X[0]
                points[index,1] = X[1]
                points[index,2] = X[2]
                index += 1

    result = {'points':points_arr}

    if normals.shape[0] > 0:
        index = 0
        for y in range(height):
            for x in range(width):
                d = depth[y,x]
                if np.isfinite(d) and d > 0.0:
                    tmp[0] = normals[0,y,x]
                    tmp[1] = normals[1,y,x]
                    tmp[2] = normals[2,y,x]
                    X[0] = R[0,0]*tmp[0] + R[1,0]*tmp[1] + R[2,0]*tmp[2]
                    X[1] = R[0,1]*tmp[0] + R[1,1]*tmp[1] + R[2,1]*tmp[2]
                    X[2] = R[0,2]*tmp[0] + R[1,2]*tmp[1] + R[2,2]*tmp[2]
                    normals_attr[index,0] = X[0]
                    normals_attr[index,1] = X[1]
                    normals_attr[index,2] = X[2]
                    index += 1

        result['normals'] = normals_attr_arr
           
    if colors.shape[0] > 0:
        index = 0
        for y in range(height):
            for x in range(width):
                d = depth[y,x]
                if np.isfinite(d) and d > 0.0:
                    colors_attr[index,0] = colors[0,y,x]
                    colors_attr[index,1] = colors[1,y,x]
                    colors_attr[index,2] = colors[2,y,x]
                    index += 1

        result['colors'] = colors_attr_arr
    
    return result



def compute_point_cloud_from_depthmap( depth, K, R, t, normals=None, colors=None ):
    """Creates a point cloud numpy array and optional normals and colors arrays

    depth: numpy.ndarray 
        2d array with depth values

    K: numpy.ndarray
        3x3 matrix with internal camera parameters

    R: numpy.ndarray
        3x3 rotation matrix

    t: numpy.ndarray
        3d translation vector

    normals: numpy.ndarray
        optional array with normal vectors

    colors: numpy.ndarray
        optional RGB image with the same dimensions as the depth map.
        The shape is (3,h,w) with type uint8

    """
    assert colors.dtype == np.uint8 if not colors is None else True

    # make sure the dims and type are ok for the depth
    if depth.dtype != np.float32:
        _depth = depth.astype(np.float32)
    else:
        _depth = depth

    if len(_depth.shape) > 2:
        _depth = _depth.squeeze()
    if len(_depth.shape) > 2:
        raise ValueError("wrong number of dimensions for depth")

    # sanity checks
    if normals is None:
        normals = np.empty((0,0,0),dtype=np.float32)
    elif normals.shape[1:] != _depth.shape:
        raise ValueError("shape mismatch: normals {0}, depth {1}".format(normals.shape, depth.shape))

    if normals.dtype != np.float32:
        _normals = normals.astype(np.float32)
    else:
        _normals = normals
        
    if colors is None:
        colors_arr = np.empty((0,0,0),dtype=np.uint8)
    else:
        colors_arr = colors
        if colors_arr.shape[1:] != _depth.shape:
            raise ValueError("shape mismatch: colors {0}, depth {1}".format(colors_arr.shape, depth.shape))

    return _compute_point_cloud_from_depthmap(_depth, _normals, colors_arr, K.astype(np.float32), R.astype(np.float32), t.astype(np.float32))

