from random import randint
import os
import torch

def saltn(n):
    range_start = 10**(n-1)
    range_end = (10**n)-1
    return randint(range_start, range_end)

def save_images_and_hspace(images, hspace, latent_dims, save_path="./outputs", print_path=True,):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(os.path.join(save_path, 'images'))
        os.makedirs(os.path.join(save_path, 'hspace'))

    for i, image in enumerate(images):
        salt = str(saltn(4))
        image_path  = os.path.join(save_path, 'images', f"image_{salt}.png" )
        hspace_path = os.path.join(save_path, 'hspace', f"hspace_{salt}.pt" )
        image.save(image_path)
        torch.save( hspace[:,i:i+latent_dims], hspace_path)
        if print_path:
            print(f"Image saved: {image_path}")
            print(f"Hspace saved: {hspace_path}")