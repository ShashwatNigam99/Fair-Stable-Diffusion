import torch
from stable_diffusion_hspace import StableDiffusionPipelineHspace
from unet_hspace import UNet2DConditionModelHSpace
from pathlib import Path
import torch.nn.functional as F
import torch.nn as nn
import os
# from clip_classification import classifier
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from torchviz import make_dot

# PATH = "/home/hice1/mnigam9/scratch/cache/stable-diffusion-v1-5"
PATH = '/common/home/zw465/test/cachedir/stable-diffusion-v1-5'
PATH = Path(PATH).expanduser()


class Classifier(torch.nn.Module):
    def __init__(self, model_name_or_path, device='cpu'):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_name_or_path).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name_or_path)
        # why are we setting this to true?
        for param in self.model.parameters():
            param.requires_grad = True
        
    def forward(self, text, images):
        inputs = self.processor(text=text, images=images, return_tensors="pt", padding=True)
        inputs.to(self.model.device)
        outputs = self.model(**inputs)
        return outputs
    
    def classify(self, outputs):
        probs = outputs.logits_per_image.softmax(dim=1)
        return probs



# binary classification loss involving computing the KL divergence between the generated distribution and the target distribution
def classify_loss(class_model, images): 
    text_classes = ["photo of a man", "photo of a woman" ]
    # prompts = ["a photo of an Asian","a photo of a caucasian", "a photo of a black person", "a photo of a latin American"]
    outputs = class_model.forward(text_classes, images)
    # outputs_race = class_model.forward(prompts, images)
    prob_dist = class_model.classify(outputs)

    prob_dist = torch.sum(prob_dist, dim=0)

    prob_dist /= torch.sum(prob_dist) 
    
    loss = prob_dist[0] * torch.log(2 * prob_dist[0]) + prob_dist[1] * torch.log(2* prob_dist[1])
    
    return loss


def setup_hspace_stable_diffusion(PATH, latent_lambda):
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
    hspace_unet.set_deltablock(latent_lambda)
    hspace_unet = hspace_unet.to("cuda")
    hspace_unet.deltablock = hspace_unet.deltablock.to("cuda").to(torch.float16)
    freeze_params(hspace_unet)
    
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
    "num_inference_steps": 10,
    "num_images_per_prompt": 8,
    "classifier_free_guidance": True
}

def get_config():
    CONFIG["latent_dims"] = 1
    if CONFIG["classifier_free_guidance"]:
        CONFIG["latent_dims"] = 2
    CONFIG["latent_lambda"] = 0.4
    return CONFIG


def freeze_params(unet):
    for param in unet.parameters():
        param.requires_grad = True # why change this to true?
    for param in unet.deltablock.parameters():
        nn.init.zeros_(param)
        param.requires_grad = True
        

def compute_kl_divergence(generated_distribution, target_distribution):
    target_distribution = target_distribution / target_distribution.sum()
    target_log_prob = torch.log(target_distribution)
    kl_div = F.kl_div(generated_distribution, target_log_prob, reduction='batchmean')
    return kl_div

if __name__=='__main__':
    config = get_config()
    hspace_pipe = setup_hspace_stable_diffusion(PATH,config["latent_lambda"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
    # class_model = Classifier("/home/hice1/mnigam9/scratch/cache/clip-vit-large-patch14", device)
    class_model = Classifier("/common/home/zw465/test/cachedir/clip-vit-large-patch14", device)

    optimizer = torch.optim.SGD(hspace_pipe.unet.deltablock.parameters(), lr=0.01, momentum=0.9)
    hspace_pipe.unet.deltablock.train()
    for i in range(100):
        optimizer.zero_grad()
        breakpoint()
        images, hspace = hspace_pipe(
            prompt = config["prompts"],
            num_inference_steps = config["num_inference_steps"],
            num_images_per_prompt = config["num_images_per_prompt"]    
        )
        
        loss = classify_loss(class_model, images.images) 
        # print(prob_dist)
        # loss = prob_dist[0] * torch.log(2 * prob_dist[0]) + prob_dist[1] * torch.log(2* prob_dist[1])
        dot = make_dot(loss, params=dict(list(hspace_pipe.unet.named_parameters())))
        dot.render('computational_graph', format='png')  # Saves the graph as a PNG image
        loss.backward()
        optimizer.step()
        print(loss)