from stable_diffusion_hspace import StableDiffusionPipelineHspace
from unet_hspace import UNet2DConditionModelHSpace
import torch
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
from pathlib import Path

PATH = "/home/hice1/mnigam9/scratch/cache/stable-diffusion-v1-4"
PATH = Path(PATH).expanduser()

def setup_hspace_stable_diffusion(PATH):
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
    # disable for old GPUs like V100
    # hspace_unet.enable_xformers_memory_efficient_attention(
    #     attention_op=MemoryEfficientAttentionFlashAttentionOp
    # )
    
    return hspace_pipe

if __name__=="__main__":
    hspace_pipe = setup_hspace_stable_diffusion(PATH)
    prompts = ["A photo of a cat", "A photo of a dog", "A photo of a bird"]
    
    images, hspace = hspace_pipe(prompts)
    
    print(images, hspace.shape)
    
    
    