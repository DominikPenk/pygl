import matplotlib.pyplot as plt

from pygl.framebuffer import FrameBuffer
from pygl.texture import Texture2D
import numpy as np

def _convert_to_numpy_arrays(data):
    if isinstance(data, np.ndarray):
        return [data]
    elif isinstance(data, FrameBuffer):
        # Download and convert all attachments
        return list(tex.download() for tex in data.attachments.values())
    elif isinstance(data, Texture2D):
        return [data.download()]
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

def display_images(images,
                   title=None,
                   size=10, 
                   **kwargs):
    """
    Display a list of images.
    """
    if not isinstance(images, (tuple, list)):
        images = [images]

    # Convert images to a list of numpy arrays
    imgs = []
    for img in images:
        imgs.extend(_convert_to_numpy_arrays(img))
    
    # Display images
    num_images = len(imgs)
    fig, axs = plt.subplots(1, 
                            num_images, 
                            figsize=(num_images * size, size),
                            squeeze=False)
    for ax, img in zip(axs.flat, imgs):
        ax.imshow(img, **kwargs)
        ax.axis('off')

    if title:
        fig.suptitle(title, fontsize=16)
    plt.show()