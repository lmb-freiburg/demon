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

def add_sources(params, dataset_files, weight, normalize=True, concatenate=False):
    """Add sources to the parameters for the multi_vi_h5_data_reader op.

    params: dict
        dict with the parameters for the multi_vi_h5_data_reader op.

    dataset_files: list of str
        List of h5 file paths to be added as sources.

    weight: float
        The sampling importance. 
        Higher values means the reader op samples more often from these files.

    normalize: bool
        If True the weight for each file will be divided by the number of files.
        If concatenate is True this parameter has no effect.

    concatenate: bool
        If True adds only a single source that contains all files.

    """
    if not 'source' in params:
        params['source'] = []

    if concatenate:
        # generate a single source with all paths
        source = {'path': ';'.join(dataset_files)}
        params['source'].append(source)

    else:
        # generate for each path a new source
        for item in dataset_files:
            w = weight
            if normalize:
                w /= len(dataset_files)

            source = {'path': item, 'weight': w}
            params['source'].append(source)

    return params
