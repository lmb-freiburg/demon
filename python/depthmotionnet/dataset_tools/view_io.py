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
from PIL import Image
from io import BytesIO
from .lz4 import lz4_uncompress, lz4_compress_HC
from .webp import webp_encode_array, webp_encode_image

from .view import View


def read_webp_image(h5_dataset):
    """Reads a dataset that stores an image compressed as webp
    
    h5_dataset : hdf5 dataset object

    Returns the image as PIL Image
    """
    data = h5_dataset[:].tobytes()
    img_bytesio = BytesIO(data)
    pil_img = Image.open(img_bytesio,'r')
    return pil_img


def write_webp_image(h5_group, image, dsname="image"):
    """Writes the image as webp to a new dataset

    h5_group: hdf5 group
        The group that shall contain the newly created dataset

    image: PIL.Image or rgb numpy array
        The image
    """
    if isinstance(image,np.ndarray):
        compressed_data = webp_encode_array(image)
    else:
        compressed_data = webp_encode_image(image)
    image_compressed = np.frombuffer(compressed_data,dtype=np.int8)
    ds = h5_group.create_dataset(dsname, data=image_compressed)
    ds.attrs['format'] = np.string_("webp")



def read_lz4half_depth(h5_dataset):
    """Reads a dataset that stores a depth map in lz4 compressed float16 format
    
    h5_dataset : hdf5 dataset object

    Returns the depth map as numpy array with float32
    """
    extents = h5_dataset.attrs['extents']
    num_pixel = extents[0]*extents[1]
    expected_size = 2*num_pixel
    data = h5_dataset[:].tobytes()
    depth_raw_data = lz4_uncompress(data,int(expected_size))
    depth = np.fromstring(depth_raw_data,dtype=np.float16)
    depth = depth.astype(np.float32)
    depth = depth.reshape((extents[0],extents[1]))
    return depth


def write_lz4half_depth(h5_group, depth, depth_metric, dsname="depth"):
    """Writes the depth as 16bit lz4 compressed char array to the given path

    h5_group: hdf5 group
        The group that shall contain the newly created dataset

    depth: numpy array with float32
    """
    assert isinstance(depth, np.ndarray), "depth must be a numpy array"
    assert depth.dtype == np.float32, "depth must be a float32 array"
    assert len(depth.shape) == 2, "depth must be a 2d array"
    assert depth_metric in ('camera_z', 'ray_length'), "depth metric must be either 'camera_z' or 'ray_length'"
    height = depth.shape[0]
    width = depth.shape[1]
    depth16 = depth.astype(np.float16)
    depth_raw_data = depth16.tobytes()
    compressed_data = lz4_compress_HC(depth_raw_data)
    depth_compressed = np.frombuffer(compressed_data,dtype=np.int8)
    ds = h5_group.create_dataset(dsname, data=depth_compressed)
    ds.attrs['format'] = np.string_("lz4half")
    ds.attrs['extents'] = np.array([height, width], dtype=np.int32)
    ds.attrs['depth_metric'] = np.string_(depth_metric)


def read_camera_params(h5_dataset):
    """Reads a dataset that stores camera params in float64
    
    h5_dataset : hdf5 dataset object

    Returns K,R,t as numpy array with float64
    """
    fx = h5_dataset[0]
    fy = h5_dataset[1]
    skew = h5_dataset[2]
    cx = h5_dataset[3]
    cy = h5_dataset[4]
    K = np.array([[fx, skew, cx],
                 [0, fy, cy],
                 [0, 0, 1]], dtype=np.float64)
    R = np.array([[h5_dataset[5], h5_dataset[8], h5_dataset[11]], 
                  [h5_dataset[6], h5_dataset[9], h5_dataset[12]], 
                  [h5_dataset[7], h5_dataset[10], h5_dataset[13]]], dtype=np.float64)
    t = np.array([h5_dataset[14], h5_dataset[15], h5_dataset[16]], dtype=np.float64)   
    return K,R,t


def write_camera_params(h5_group, K, R, t, dsname="camera"):
    """Writes the camera params as float64 to the given path

    h5_group: hdf5 group
        The group that shall contain the newly created dataset

    K, R, t: numpy array with float64
    """
    data = np.array([K[0,0], K[1,1], K[0,1], K[0,2], K[1,2], 
                    R[0,0], R[1,0], R[2,0], R[0,1], R[1,1], R[2,1], R[0,2], R[1,2], R[2,2], 
                    t[0], t[1], t[2]], dtype=np.float64)
    ds = h5_group.create_dataset(dsname, data=data)
    ds.attrs['format'] = "pinhole".encode('ascii')


def read_view(h5_group):
    """Reads the view group and returns it as a View tuple
    
    h5_group: hdf5 group
        The group for reading the view

    Returns the View tuple
    """
    img = read_webp_image(h5_group['image'])
    depth = read_lz4half_depth(h5_group['depth'])
    depth_metric = h5_group['depth'].attrs['depth_metric'].decode('ascii')
    K_arr,R_arr,t_arr = read_camera_params(h5_group['camera'])
    return View(image=img, depth=depth, depth_metric=depth_metric, K=K_arr, R=R_arr, t=t_arr)


def write_view(h5_group, view):
    """Writes the View tuple to the group

    h5_group: hdf5 group
        The group for storing the view

    view: View namedtuple
        The tuple storing the view
    
    """
    for ds in ('image', 'depth', 'camera'):
        if ds in h5_group:
            del h5_group[ds]

    write_webp_image(h5_group, view.image)
    write_lz4half_depth(h5_group, view.depth, view.depth_metric)
    write_camera_params(h5_group, view.K, view.R, view.t)


