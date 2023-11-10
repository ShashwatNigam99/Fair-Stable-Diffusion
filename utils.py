from PIL import Image 
from random import randint
import os

def salt(n):
    range_start = 10**(n-1)
    range_end = (10**n)-1
    return randint(range_start, range_end)

def save_images(images, save_path="./outputs/images/", prefix="image", print_path=True):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for image in images:
        image_path = os.path.join(save_path, f"{prefix}_{str(salt(4))}.png" )
        image.save(image_path)
        if print_path:
            print(f"Image saved: {image_path}")