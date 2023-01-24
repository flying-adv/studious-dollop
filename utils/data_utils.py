"""
Code adopted from pix2pixHD:
https://github.com/NVIDIA/pix2pixHD/blob/master/data/image_folder.py
"""
import os
from tqdm import tqdm

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


# def make_dataset(dir):
#     images = []
#     assert os.path.isdir(dir), '%s is not a valid directory' % dir
#     for root, _, fnames in sorted(os.walk(dir)):
#         for fname in fnames:
#             if is_image_file(fname):
#                 path = os.path.join(root, fname)
#                 images.append(path)
#     return images

def make_dataset(dir):
    images = []
    for path in sorted(os.listdir(dir)):
        images.append(os.path.join(dir , path))
    return images

# def make_dataset(dir):
#     images = []
#     #print(dir)
#     for data in tqdm(os.listdir(dir)):
#         #print(data)
#         data = os.path.join(dir , data)
#         for files in os.listdir(data):
#             #if is_image_file(files):
#             path = os.path.join(data , files)
#             images.append(path)
#     print(len(images))
#     return images    
