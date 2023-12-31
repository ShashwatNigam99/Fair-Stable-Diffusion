from stable_diffusion_hspace import StableDiffusionPipelineHspace
from unet_hspace import UNet2DConditionModelHSpace
import torch
from pathlib import Path
from utils import save_images_and_hspace
import time
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
    print('here')
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
    "prompts": ["Photo portrait of a doctor", "Photo portrait of a teacher", "Photo portrait of a lawyer"],    
    "num_inference_steps": 50,
    "num_images_per_prompt": 5,
    "classifier_free_guidance": True
}

def get_config():
    CONFIG["latent_dims"] = 1
    if CONFIG["classifier_free_guidance"]:
        CONFIG["latent_dims"] = 2    
    return CONFIG

if __name__=="__main__":
    hspace_pipe = setup_hspace_stable_diffusion(PATH)
    config = get_config()
    # images here is of type StableDiffusionPipelineOutput. To access images, use images.images
    start = time.time()
    images, hspace = hspace_pipe(
        prompt = config["prompts"],
        num_inference_steps = config["num_inference_steps"],
        num_images_per_prompt = config["num_images_per_prompt"]    
    )
    end = time.time()
    print("Time taken: ", end-start)
    
    # breakpoint()
    print("CONFIG: ", config)
    print("Number of images: ", len(images.images))
    print("Hspace dimension: ", hspace.shape)
    
    save_images_and_hspace(
        images.images, 
        hspace, 
        config["latent_dims"],
        save_path="./outputs/",
        print_path=False
    )
    
    
    