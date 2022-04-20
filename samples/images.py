import os

import numpy as np
import skimage.io as skio

def _get_image_dir():
    image_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'images')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    return image_dir

def _get_image(name, url):
    image_dir = _get_image_dir()
    image_path = os.path.join(image_dir, name)
    image_path = os.path.abspath(image_path)
    if not os.path.exists(image_path):
        import requests
        print(f'Downloading image: {name} (Destination: {image_path})')
        r = requests.get(url)
        with open(image_path, 'wb') as f:
            f.write(r.content)
    img = skio.imread(image_path)
    if img.shape[-1] == 3:
        img = np.concatenate([
            img,
            np.full(img.shape[:2] + (1,), 255, dtype=np.uint8)
        ], axis=-1)
    return img

def lena():
    import requests
    URL = r"https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png"
    return _get_image('lena.png', URL)
