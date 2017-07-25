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
import sys
import math
import pickle
import argparse
import itertools
import h5py
from multiprocessing import Pool
datasets_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(datasets_dir, '..', 'python'))

from depthmotionnet.dataset_tools.sun3d_utils import *
from depthmotionnet.dataset_tools.view_tools import *
from depthmotionnet.dataset_tools.view_io import *


def create_train_file(outfile, sun3d_data_path, seq_name, baseline_range, seq_sharpness_dict):
    """Creates a h5 file with training samples with a specific baseline range

    outfile: str
        Output file

    sun3d_data_path: str
        The path to the sun3d data directory

    seq_name: str
        sequence name

    baseline_range: tuple(float, float)
        Minimum and maximum baseline

    seq_sharpness_dict: dict
        Dictionary with the sharpness score of all sequences.
        key: str with sequence name
        value: numpy.ndarray with sharpness scores

    """
    created_groups = 0
    with h5py.File(outfile,'w') as f:
        created_groups += create_samples_from_sequence(f, sun3d_data_path, seq_name, baseline_range, seq_sharpness_dict[seq_name])
    return created_groups


def merge_h5files(outfile, files):
    """Merges multiple h5 files into a new file and removes the input files afterwards.

    outfile: str
        Output file

    files: list of str
        List of files to merge
    """
    with h5py.File(outfile,'w') as dst:
        for f in files:
            print('copy', f, 'to', outfile)
            with h5py.File(f,'r') as src:
                for group_name in src:
                    src.copy(source=group_name, dest=dst)
    for f in files:
        os.remove(f)




def main():

    print(
"""================================================================================

 This script runs for about 1 day on a computer with 16 threads and requires
 up to 50GB of disk space in the output directory!

================================================================================""")

    parser = argparse.ArgumentParser(description="Generates the sun3d training datasets.")
    parser.add_argument("--sun3d_path", type=str, required=True, help="The path to the sun3d data directory")
    parser.add_argument("--outputdir", type=str, default='training_data', help="Output directory for the generated h5 files")
    parser.add_argument("--threads", type=int, default=16, help="Number of threads")

    args = None
    try:
        args = parser.parse_args()
        print(args)
    except:
        return 1

    sun3d_data_path = args.sun3d_path
    outputdir = args.outputdir
    os.makedirs(outputdir, exist_ok=True)
    threads = args.threads

    # read txt file with the train sequence names
    with open('sun3d_train_sequences.txt', 'r') as f:
        sequences = f.read().splitlines()

    # compute the sharpness scores for all sequences and images
    if os.path.isfile('sun3d_seq_sharpness_dict.pkl'):
        print('Reading sequence sharpness file seq_sharpness_dict.pkl')
        with open('sun3d_seq_sharpness_dict.pkl','rb') as f:
            seq_sharpness_dict = pickle.load(f)
    else:
        print('Computing sharpness for all images. This could take a while.')
        with Pool(threads) as pool:
            args = [(sun3d_data_path, seq,) for seq in sequences]    
            sequence_sharpness = pool.starmap(compute_sharpness, args, chunksize=1)

        seq_sharpness_dict = dict(zip(sequences, sequence_sharpness))

        with open('sun3d_seq_sharpness_dict.pkl','wb') as f:
            pickle.dump(seq_sharpness_dict, f)


    # baseline ranges from 1cm-10cm to 1.6m-inf
    baseline_ranges = [(0.01,0.10), (0.10,0.20), (0.20,0.40), (0.40,0.80), (0.80,1.60), (1.60, float('inf'))]

    with Pool(threads) as pool:

        # create temporary h5 files for each baseline and sequence combination
        baseline_range_files_dict = {b:[] for b in baseline_ranges}
        args = []
        for i, base_range_seq_name in enumerate(itertools.product(baseline_ranges, sequences)):
            base_range, seq_name = base_range_seq_name
            #print(base_range, seq_name)
            outfile = os.path.join(outputdir,"{0}.h5".format(i))
            args.append((outfile, sun3d_data_path, seq_name, base_range, seq_sharpness_dict))
            baseline_range_files_dict[base_range].append(outfile)

        created_groups = pool.starmap(create_train_file, args, chunksize=1)

    # merge temporary files by creating one file per baseline range
    for base_range in baseline_ranges:
        outfile = os.path.join(outputdir, 'sun3d_train_{0}m_to_{1}m.h5'.format(*base_range))
        merge_h5files(outfile, baseline_range_files_dict[base_range])


    print('created', sum(created_groups), 'groups')

    return 0


    
    

if __name__ == "__main__":
    sys.exit(main())

