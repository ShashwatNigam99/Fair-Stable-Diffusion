# pipeline for fair image generation
# need to install semdiffuser using pip install git+https://github.com/ml-research/semantic-image-editing.git


import torch
from semdiffusers import SemanticEditPipeline
from PIL import Image
import random
import os

def mp(p):
    # if p is like a/b/c/d.png, then only make a/b/c/
    first_dot = p.find('.')
    last_slash = p.rfind('/')
    if first_dot < last_slash:
        assert ValueError('Input path seems like a/b.c/d/g, which is not allowed :(')
    p_new = p[:last_slash] + '/'
    if not os.path.exists(p_new):
        os.makedirs(p_new)


def gen_occupation_face(occupation='firefighter', bias=['male person', 'female person'], seed=0, num_images_per_prompt=1, output_name=None):

    device='cuda'
    pipe = SemanticEditPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
    ).to(device)
    gen = torch.Generator(device=device)


    prompt = f'A photo of the face of a {occupation}'
    # input image
    gen.manual_seed(seed)
    
    edit = ['male person', 'female person']
    
    # enhance one concept and weakness the others
    n_properties = len(bias)
    reverse_editing_direction = [False] * n_properties
    enhance_idx = random.randint(0, n_properties-1)
    print('ENHANCE', enhance_idx)
    reverse_editing_direction[enhance_idx] = True
    
    edit_warmup_steps = [15] * n_properties
    edit_guidance_scale = [3] * n_properties
    edit_threshold = [0.9] * n_properties
    edit_weights = [1] * n_properties
    
    

    gen.manual_seed(seed)
    out = pipe(prompt=prompt, generator=gen, num_images_per_prompt=num_images_per_prompt, guidance_scale=7,
            editing_prompt=edit,       # Concepts to apply 
            reverse_editing_direction=reverse_editing_direction, # Direction of guidance i.e. increase cars, decrease crowd, and add sunset
            edit_warmup_steps=edit_warmup_steps, # Warmup period for each concept
            edit_guidance_scale=edit_guidance_scale, # Guidance scale for each concept
            edit_threshold=edit_threshold, # Threshold for each concept. Note that positive guidance needs negative thresholds and vice versa
            edit_momentum_scale=0.5, # Momentum scale that will be added to the latent guidance
            edit_beta1=0.6, # Momentum beta
            edit_weights=edit_weights # Weights of the individual concepts against each other
            )
    
    images = out.images
    
    # save output
    mp(f'out/{occupation}/')
    for i, img in enumerate(images):
        img.save(f'out/{occupation}/{output_name}.png')
        print('IMAGE OUTPUT TO:', f'out/{occupation}/{output_name}.png')

# test

if __name__ == '__main__':
    
    for i in range(10):
        gen_occupation_face('cook', seed=i, output_name=f'{i}')
        
        
    
    
    
    
    
    
    
    

