from stable_diffusion_hspace import StableDiffusionPipelineHspace
from unet_hspace import UNet2DConditionModelHSpace
import torch
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
from pathlib import Path
from utils import save_images

PATH = "/home/hice1/mnigam9/scratch/cache/stable-diffusion-v1-4"
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
        torch_dtype=torch.float16 
    )
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

if __name__=="__main__":
    hspace_pipe = setup_hspace_stable_diffusion(PATH)
    prompts = ["A photo of a cat", "A photo of a dog", "A photo of a bird"]
    
    # images here is of type StableDiffusionPipelineOutput. To access images, use images.images
    images, hspace = hspace_pipe(
        prompt = prompts,
        num_inference_steps = 20    
    )
    
    breakpoint()
    
    save_images(images.images, save_path="./outputs/images/", prefix="image", print_path=True)
    
    
    
    