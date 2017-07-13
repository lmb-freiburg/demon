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
import os
import tensorflow
from .helpers import *

# try to import the multivih5datareaderop from the 'build' directory
if 'MULTIVIH5DATAREADEROP_LIB' in os.environ:
    _readerlib_path = os.environ['MULTIVIH5DATAREADEROP_LIB']
else:
    _readerlib_path = os.path.abspath(os.path.join(os.path.split(__file__)[0], '..', '..', '..', 'build','multivih5datareaderop', 'multivih5datareaderop.so'))

readerlib = None
multi_vi_h5_data_reader = None
if os.path.isfile(_readerlib_path):
    readerlib = tensorflow.load_op_library(_readerlib_path)
    print('Using {0}'.format(_readerlib_path))
    multi_vi_h5_data_reader = readerlib.multi_vi_h5_data_reader

