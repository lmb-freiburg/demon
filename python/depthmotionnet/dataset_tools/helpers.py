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


