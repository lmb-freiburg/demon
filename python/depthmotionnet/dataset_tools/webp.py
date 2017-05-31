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
from ctypes import *
from PIL import Image
import numpy as np
import os

# try the version used by the multivih5datareaderop first
try:
    _lib_path = os.path.abspath(os.path.join(os.path.split(__file__)[0], '..', '..', '..', 'build','webp','src','webp-build', 'src', '.libs', 'libwebp.so'))
    libwebp = CDLL(_lib_path)
except:
    # try system version
    try:
        libwebp = CDLL('libwebp.so')
    except:
        raise RuntimeError('Cannot load libwebp.so')

def webp_encode_array(array, quality=90.0):
    """encode the array as webp and return as bytes.

    array: uint8 numpy array
        array with the following shape [height,width,3] or [3,height,width]

    Returns the compressed bytes array or None on error
    """
    assert isinstance(array, np.ndarray), "array must be a numpy array"
    assert array.dtype == np.uint8, "array must be a uint8 array"
    assert len(array.shape) == 3, "array must be a 3d array"
    assert array.shape[0] == 3 or array.shape[-1] == 3, "array must have 3 color channels"
    
    if array.shape[0] != array.shape[-1] and array.shape[0] == 3:
        array_rgb = array.transpose([2,0,1])
    else:
        array_rgb = array
    data = array_rgb.tobytes()

    width = c_int(array_rgb.shape[1])
    height = c_int(array_rgb.shape[0])
    stride = c_int(array_rgb.shape[1]*3)
    output = POINTER(c_char)()
    size = libwebp.WebPEncodeRGB(data, width, height, stride, c_float(quality), pointer(output))
    if size == 0:
        return None

    webp_img = output[:size]
    libwebp.WebPFree(output)
    # libc.free(output)
    return webp_img

    


def webp_encode_image(image):
    """encode the image as webp and return as bytes

    image: PIL.Image
        Image to encode
    """
    arr = np.array(image)
    return webp_encode_array(arr)
