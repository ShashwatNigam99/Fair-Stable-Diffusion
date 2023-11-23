import torch
from stable_diffusion_hspace import StableDiffusionPipelineHspace
from unet_hspace import UNet2DConditionModelHSpace
from pathlib import Path
import torch.nn.functional as F
import os
# from clip_classification import classifier
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from clip_classification import classify
PATH = "stable-diffusion-v1-5"
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
    hspace_unet.set_deltablock()
    hspace_unet = hspace_unet.to("cuda")
    hspace_unet.deltablock = hspace_unet.deltablock.to("cuda").to(torch.float16)

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
    "prompts": ["Photo portrait of a doctor"],    
    "num_inference_steps": 50,
    "num_images_per_prompt": 18,
    "classifier_free_guidance": True
}

def get_config():
    CONFIG["latent_dims"] = 1
    if CONFIG["classifier_free_guidance"]:
        CONFIG["latent_dims"] = 2    
    return CONFIG


def freeze_params(unet):
    for param in unet.parameters():
        param.requires_grad = True
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

    optimizer = torch.optim.SGD(hspace_pipe.unet.deltablock.parameters(), lr=10000, momentum=0.9)
    hspace_pipe.unet.deltablock.train()
    for i in range(100):
        optimizer.zero_grad()
        images, hspace = hspace_pipe(
            prompt = config["prompts"],
            num_inference_steps = config["num_inference_steps"],
            num_images_per_prompt = config["num_images_per_prompt"]    
        )
        # images_path = "outputs2/images/"
        # os.makedirs(images_path, exist_ok=True)

        # hh = 0
        # for h in images.images:
        #     h.save(f'{images_path}/{hh}.png')
        #     hh += 1
        # print the iamges into a folder
        
        
        prob_dist = classify(images.images) # should take path of images as input
        # uniform_dist = torch.ones(1, 2).to("cuda") * 0.5
        # torch.sum(prob_dist).backward()
        # optimizer.step()
        '''
        {'Man': 0.8387096774193549, 'Woman': 0.16129032258064516} {'white': 0.7419354838709677, 'latino hispanic': 0.0967741935483871, 'asian': 0.06451612903225806, 'middle eastern': 0.0967741935483871}
        '''
        # print(prob_dist)
        # prob_dist = torch.tensor(prob_dist).to("cuda")
        # uniform_dist = torch.ones(1, 2).to("cuda") * 0.5
        loss = prob_dist[0] * torch.log(2 * prob_dist[0]) + prob_dist[1] * torch.log(2* prob_dist[1])
        # kl_divergence = compute_kl_divergence(prob_dist, uniform_dist)
        
        loss.backward()
        optimizer.step()
        for param,name in zip(hspace_pipe.unet.deltablock.parameters(), hspace_pipe.unet.deltablock.named_parameters()):
            print(name, param)
        print(loss)