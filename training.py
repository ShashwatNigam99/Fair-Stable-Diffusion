import torch
from stable_diffusion_hspace import StableDiffusionPipelineHspace
from unet_hspace import UNet2DConditionModelHSpace
from pathlib import Path
import torch.nn.functional as F
import os
from classify import classifier

PATH = "/home/okara7/Desktop/Fair-Stable-Diffusion/stable-diffusion-v1-5"
PATH = Path(PATH).expanduser()

def setup_hspace_stable_diffusion(PATH):
    """
    Sets up a stable diffusion pipeline for HSpace using a pre-trained UNet2DConditionModelHSpace model.

    Args:
        PATH (str): The path to the directory containing the pre-trained model.

    Returns:
        A StableDiffusionPipelineHspace object configured with the pre-trained UNet2DConditionModelHSpace model.
    """
    
    hspace_unet = UNet2DConditionModelHSpace.from_pretrained(
        PATH, 
        subfolder = "unet",
        torch_dtype=torch.float16,
        strict=False
    )
    # print('here')
    # hspace_unet.set_deltablock()
    # hspace_unet = hspace_unet.to("cuda")
    # hspace_unet.deltablock = hspace_unet.deltablock.to("cuda").to(torch.float16)

    hspace_pipe = StableDiffusionPipelineHspace.from_pretrained(
        PATH, 
        torch_dtype = torch.float16, 
        use_safetensors = True,
        safety_checker = None,
        unet = hspace_unet
    ).to("cuda")
    # https://huggingface.co/docs/diffusers/optimization/memory
    hspace_pipe.enable_vae_slicing()
    
    # https://huggingface.co/docs/diffusers/v0.22.3/en/api/models/overview#diffusers.ModelMixin.enable_xformers_memory_efficient_attention
    # NOTE: disable for old GPUs like V100, RTX_6000
    # hspace_unet.enable_xformers_memory_efficient_attention(
    #     attention_op=MemoryEfficientAttentionFlashAttentionOp
    # )
    
    return hspace_pipe

CONFIG = {
    "prompts": ["Photo portrait of a doctor", "Photo portrait of a teacher", "Photo portrait of a lawyer"],    
    "num_inference_steps": 50,
    "num_images_per_prompt": 2,
    "classifier_free_guidance": True
}

def get_config():
    CONFIG["latent_dims"] = 1
    if CONFIG["classifier_free_guidance"]:
        CONFIG["latent_dims"] = 2    
    return CONFIG


def freeze_params(unet):
    for param in unet.parameters():
        param.requires_grad = False
    for param in unet.deltablock.parameters():
        param.requires_grad = True

def compute_kl_divergence(generated_distribution, target_distribution):
    target_distribution = target_distribution / target_distribution.sum()
    target_log_prob = torch.log(target_distribution)
    kl_div = F.kl_div(generated_distribution, target_log_prob, reduction='batchmean')
    return kl_div

if __name__=='__main__':
    
    hspace_pipe = setup_hspace_stable_diffusion(PATH)
    config = get_config()

    optimizer = torch.optim.SGD(hspace_pipe.unet.parameters(), lr=0.001, momentum=0.9)
    
    for i in range(100):
        optimizer.zero_grad()
        images, hspace = hspace_pipe(
            prompt = config["prompts"],
            num_inference_steps = config["num_inference_steps"],
            num_images_per_prompt = config["num_images_per_prompt"]    
        )
        images_path = '/home/okara7/Desktop/Fair-Stable-Diffusion/outputs/tmp'
        os.makedirs(images_path, exist_ok=True)

        hh = 0
        for h in images.images:
            h.save(f'{images_path}/{hh}.png')
            hh += 1
        # print the iamges into a folder
        
        
        prob_dist = classifier(images_path) # should take path of images as input
        '''
        {'Man': 0.8387096774193549, 'Woman': 0.16129032258064516} {'white': 0.7419354838709677, 'latino hispanic': 0.0967741935483871, 'asian': 0.06451612903225806, 'middle eastern': 0.0967741935483871}
        '''
        print(prob_dist)
        
        uniform_dist = torch.ones(images.shape[0], 2) * 0.5
        kl_divergence = compute_kl_divergence(prob_dist, uniform_dist)
        kl_divergence.backward()
        optimizer.step()
        print("KL divergence: ", kl_divergence)