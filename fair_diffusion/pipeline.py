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



def gen_occupation_face(occupation='firefighter', bias='gender', seed=0, num_images_per_prompt=1, output_name=None, raw=False):

    device='cuda'
    pipe = SemanticEditPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
    ).to(device)
    gen = torch.Generator(device=device)


    prompt = f'A photo of the face of a {occupation}'
    # input image
    gen.manual_seed(seed)
    
    
    if bias == 'gender':
        edit = ['male person', 'female person']
    elif bias == 'color':
        edit = ['white person', 'black person']
    else:
        raise KeyboardInterrupt('unknow bias')
    
    
    # enhance one concept and weakness the others
    n_properties = len(edit)
    reverse_editing_direction = [False] * n_properties
    enhance_idx = random.randint(0, n_properties-1)
    print('ENHANCE', enhance_idx)
    reverse_editing_direction[enhance_idx] = True
    
    edit_warmup_steps = [15] * n_properties
    edit_guidance_scale = [3] * n_properties
    edit_threshold = [0.9] * n_properties
    edit_weights = [1] * n_properties
    
    

    gen.manual_seed(seed)
    
    
    if not raw:
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
    else: # sd-v1.5 baseline, has bias
        out = pipe(prompt=prompt, generator=gen, num_images_per_prompt=num_images_per_prompt, guidance_scale=7)

    
    images = out.images
    
    # save output
    method = 'fairdiff' if not raw else 'stablediff'
    mp(f'out/{occupation}/{method}/')
    for i, img in enumerate(images):
        img.save(f'out/{occupation}/{method}/{output_name}.png')
        print('IMAGE OUTPUT TO:', f'out/{occupation}/{method}/{output_name}.png')

# test
if __name__ == '__main__':
    
    
    occupation_list = [
        ('firefighter', 'gender'),
        ('accountant', 'color'),
        ('CEO', 'gender'),
        ('cleaner', 'gender'),
        ('dentist', 'color'),
        ('manager', 'gender'),
        ('professor', 'gender'),
        ('writer', 'gender'),
        ('mover', 'color'),
        ('manicurist', 'color')
    ]
    
    num_for_each_prompt = 100
    
    for occupation, bias in occupation_list:
        for i in range(num_for_each_prompt):
            gen_occupation_face('occupation', bias=bias, seed=i, output_name=f'{i}', raw=False)
            gen_occupation_face('occupation', bias=bias, seed=i, output_name=f'{i}', raw=True)
        
        
    
    
    
    
    
    
    
    

