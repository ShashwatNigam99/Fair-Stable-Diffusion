from PIL import Image

import requests

from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"

# image = Image.open(requests.get(url, stream=True).raw)
# Path to your local image
image_path = 'Fair-Stable-Diffusion/outputs/images/image_1264.png'  # Replace with your local image path

# Open the image
image = Image.open(image_path)
promts = ["a photo of an Asian person","a photo of a caucasian", "a photo of a black person", "a photo of a latin American person"]
inputs = processor(text=promts, images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)

logits_per_image = outputs.logits_per_image  # this is the image-text similarity score

probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

print(probs)