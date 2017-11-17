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
import os

# try the version used by the multivih5datareaderop first
try:
    _lib_path = os.path.abspath(os.path.join(os.path.split(__file__)[0], '..', '..', '..', 'build','lz4','src','lz4-build','contrib', 'cmake_unofficial', 'liblz4.so'))
    liblz4 = CDLL(_lib_path)
except:
    # try system version
    try:
        liblz4 = CDLL('liblz4.so')
    except:
        raise RuntimeError('Cannot load liblz4.so')


def lz4_uncompress(input_data, expected_decompressed_size):
    """decompresses the LZ4 compressed data
    
    input_data: bytes
        byte string of the input data

    expected_decompressed_size: int
        size of the decompressed output data

    returns the decompressed data as bytes or None on error
    """
    assert isinstance(input_data,bytes), "input_data must be of type bytes"
    assert isinstance(expected_decompressed_size,int), "expected_decompressed_size must be of type int"

    dst_buf = create_string_buffer(expected_decompressed_size)
    status = liblz4.LZ4_decompress_safe(input_data,dst_buf,len(input_data),expected_decompressed_size)
    if status != expected_decompressed_size:
        return None
    else:
        return dst_buf.raw



def lz4_compress_bound(input_size):
    """Returns the maximum size needed for compressing data with the given input_size"""
    assert isinstance(input_size,int), "input_size must be of type int"
    
    result = liblz4.LZ4_compressBound(c_int(input_size))
    return result



def lz4_compress_HC(src):
    """Compresses the input bytes with LZ4 high compression algorithm.

    Returns the compressed bytes array or an empty array on error
    """
    assert isinstance(src,bytes), "src must be of type bytes"
    max_compressed_size = lz4_compress_bound(len(src))
    dst_buf = create_string_buffer(max_compressed_size)
    # written_size = liblz4.LZ4_compress_HC(src, dst_buf, len(src), max_compressed_size, c_int(0)) # new signature. TODO update liblz4
    written_size = liblz4.LZ4_compressHC(src, dst_buf, len(src))
    return dst_buf.raw[:written_size]
    
