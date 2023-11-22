#write a custom classifiier module using clip for classification
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
from pathlib import Path

class Classifier(torch.nn.Module):
    def __init__(self, model_name_or_path, device='cpu'):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_name_or_path).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name_or_path)

    def forward(self, text, images):
        inputs = self.processor(text=text, images=images, return_tensors="pt", padding=True)
        inputs.to(self.model.device)
        outputs = self.model(**inputs)
        return outputs
    
    def classify(self, outputs):
        probs = outputs.logits_per_image.softmax(dim=1)
        return probs.cpu().detach().numpy()
    
OUTPUTS = "/home/hice1/mnigam9/scratch/cache/deltahspace"
OUTPUTS = Path(OUTPUTS).expanduser()

def load_image(image_path):
    return Image.open(str(img))

if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Classifier("/home/hice1/mnigam9/scratch/cache/clip-vit-large-patch14", device)

    image_names, images = [], []
    for img in OUTPUTS.iterdir():
        image_names.append(str(img))
        images.append(load_image(img))

    text_classes = ["photo of a man", "photo of a woman" ]
    
    outputs = model.forward(text_classes, images)
    probs = model.classify(outputs)
    
    print("image", text_classes)
    for i, img in enumerate(image_names):
        print(img.split('/')[-1], probs[i])