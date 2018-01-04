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
from scipy.ndimage.filters import laplace


def measure_sharpness(img):
    """Measures the sharpeness of an image using the variance of the laplacian

    img: PIL.Image

    Returns the variance of the laplacian. Higher values mean a sharper image
    """
    img_gray = np.array(img.convert('L'), dtype=np.float32)
    return np.var(laplace(img_gray))


def concat_images_vertical(images):
    """Concatenates a list of PIL.Image in vertical direction

    images: list of PIL.Image

    Returns the concatenated image
    """
    total_height = 0
    total_width = 0
    for img in images:
        total_width = max(total_width, img.size[0])
        total_height += img.size[1]
    result = Image.new('RGB',(total_width,total_height))
    ypos = 0
    for img in images:
        result.paste(img,(0,ypos))
        ypos += img.size[1]
    return result


def concat_images_horizontal(images):
    """Concatenates a list of PIL.Image in horizontal direction

    images: list of PIL.Image

    Returns the concatenated image
    """
    total_height = 0
    total_width = 0
    for img in images:
        total_height = max(total_height, img.size[1])
        total_width += img.size[0]
    result = Image.new('RGB',(total_width,total_height))
    xpos = 0
    for img in images:
        result.paste(img,(xpos,0))
        xpos += img.size[0]
    return result


def safe_crop_image(image, box, fill_value):
    """crops an image and adds a border if necessary
    
    image: PIL.Image

    box: 4 tuple
        (x0,y0,x1,y1) tuple

    fill_value: color value, scalar or tuple

    Returns the cropped image
    """
    x0, y0, x1, y1 = box
    if x0 >=0 and y0 >= 0 and x1 < image.width and y1 < image.height:
        return image.crop(box)
    else:
        crop_width = x1-x0
        crop_height = y1-y0
        tmp = Image.new(image.mode, (crop_width, crop_height), fill_value)
        safe_box = (
            max(0,min(x0,image.width-1)),
            max(0,min(y0,image.height-1)),
            max(0,min(x1,image.width)),
            max(0,min(y1,image.height)),
            )
        img_crop = image.crop(safe_box)
        x = -x0 if x0 < 0 else 0
        y = -y0 if y0 < 0 else 0
        tmp.paste(image, (x,y))
        return tmp


def safe_crop_array2d(arr, box, fill_value):
    """crops an array and adds a border if necessary
    
    arr: numpy.ndarray with 2 dims

    box: 4 tuple
        (x0,y0,x1,y1) tuple. x is the column and y is the row!

    fill_value: scalar

    Returns the cropped array
    """
    x0, y0, x1, y1 = box
    if x0 >=0 and y0 >= 0 and x1 < arr.shape[1] and y1 < arr.shape[0]:
        return arr[y0:y1,x0:x1]
    else:
        crop_width = x1-x0
        crop_height = y1-y0
        tmp = np.full((crop_height, crop_width), fill_value, dtype=arr.dtype)
        safe_box = (
            max(0,min(x0,arr.shape[1]-1)),
            max(0,min(y0,arr.shape[0]-1)),
            max(0,min(x1,arr.shape[1])),
            max(0,min(y1,arr.shape[0])),
            )
        x = -x0 if x0 < 0 else 0
        y = -y0 if y0 < 0 else 0
        safe_width = safe_box[2]-safe_box[0]
        safe_height = safe_box[3]-safe_box[1]
        tmp[y:y+safe_height,x:x+safe_width] = arr[safe_box[1]:safe_box[3],safe_box[0]:safe_box[2]]
        return tmp

