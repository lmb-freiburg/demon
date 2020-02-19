"""
This is an optional script, to load the h5 datasets into the filesystem as .png and .txt files, if preferred over h5.
To use, once either the download_traindata.sh or download_testdata.sh scripts have completed, run this script from
inside the directory containing the .h5 files. This will extract and decompress the h5 dataset into the filesystem
"""
from PIL import Image
from io import BytesIO
import lz4.block
import numpy as np
import h5py
import cv2
import os


def main():
    h5_filenames = [item for item in os.listdir('.') if item[-3:] == '.h5']
    for h5_filename in h5_filenames:
        h5_dir = h5_filename[:-3]
        os.makedirs(h5_dir, exist_ok=True)
        with h5py.File(h5_filename, 'r') as file:
            save_to_filesystem(h5_dir, file)


def save_to_filesystem(h5_dir, element):
    for key, value in element.items():
        if type(value) is h5py.Group:
            os.makedirs(os.path.join(h5_dir, key), exist_ok=True)
            save_to_filesystem(os.path.join(h5_dir, key), value)
        elif type(value) is h5py.Dataset:
            if value.dtype == np.uint8 or value.dtype == np.int8:
                uncompressed_size = 480*640*2 if key == 'depth' else 480*640*3
                image_dtype = np.uint16 if key == 'depth' else np.uint8
                compressed_bytes = np.array(value).tobytes()
                if key == 'depth':
                    decompressed_bytes = lz4.block.decompress(compressed_bytes, uncompressed_size=uncompressed_size)
                    cv2_image = np.frombuffer(decompressed_bytes, dtype=image_dtype).reshape((480, 640, -1))
                else:
                    img_bytesio = BytesIO(compressed_bytes)
                    pil_img = Image.open(img_bytesio, 'r')
                    cv2_image = np.array(pil_img.convert('RGB'))
                cv2.imwrite(os.path.join(h5_dir, key) + '.png', cv2_image)
            elif value.dtype == np.float32 or value.dtype == np.float64:
                with open(os.path.join(h5_dir, key) + '.txt', 'w+') as vector_file:
                    vector_file.write(str(np.asarray(value).tolist())[1:-1])


if __name__ == '__main__':
    main()
