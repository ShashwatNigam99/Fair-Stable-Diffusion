#write a custom classifiier module using clip for classification
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
from pathlib import Path
import numpy as np
class Classifier(torch.nn.Module):
    def __init__(self, model_name_or_path, device='cpu'):
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_name_or_path).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name_or_path)
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
    
OUTPUTS = "outputs2/images"
OUTPUTS = Path(OUTPUTS).expanduser()

def load_image(image_path):
    return Image.open(str(image_path))

def classify(images): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Classifier("clip-vit-base-patch32", device)
    # path = Path(image_path)

    # image_names, images = [], []
    # for img in path.rglob('*'):
    #     image_names.append(img)
    #     images.append(load_image(img))

    text_classes = ["photo of a man", "photo of a woman" ]
    prompts = ["a photo of an Asian","a photo of a caucasian", "a photo of a black person", "a photo of a latin American"]
    outputs = model.forward(text_classes, images)
    outputs_race = model.forward(prompts, images)
    probs_gender = model.classify(outputs)
    probs_race = model.classify(outputs_race)
    # print("image", text_classes)
    # for i, img in enumerate(image_names):
    #     print(img.split('/')[-1], probs[i])
    #     print(img.split('/')[-1], probs_race[i])
    # gender_labels = ['Man', 'Woman']
    # race_labels = ['asian', 'white', 'black', 'latino']

    # # Initialize counters for each attribute
    # gender_counts = {label: 0 for label in gender_labels}
    # race_counts = {label: 0 for label in race_labels}

    # # Iterate over each image's probabilities and count the occurrences of each attribute
    # for g_prob, r_prob in zip(probs_gender, probs_race):
    #     # Get the index of the highest probability for gender and race
    #     max_g_index = np.argmax(g_prob)
    #     max_r_index = np.argmax(r_prob)

    #     # Increment the corresponding attribute count
    #     gender_counts[gender_labels[max_g_index]] += 1
    #     race_counts[race_labels[max_r_index]] += 1

    # # Calculate the distribution percentages
    # total_images = len(probs_gender)
    # gender_distribution = {k: v / total_images for k, v in gender_counts.items()}
    # race_distribution = {k: v / total_images for k, v in race_counts.items()}
    # return gender_distribution, race_distribution

    gender_labels = ['Man', 'Woman']
    race_labels = ['asian', 'white', 'black', 'latino']
    # print(probs_gender)
    probs_gender = torch.sum(probs_gender, dim=0)
    # print(probs_gender)
    probs_gender /= torch.sum(probs_gender) 
    # input(probs_gender)

    # Use PyTorch operations to find the max indices
    # _, max_g_indices = torch.max(probs_gender, 1)
    # _, max_r_indices = torch.max(probs_race, 1)

    # Initialize tensors to store counts
    # gender_counts = torch.zeros(len(gender_labels)).to(device)
    # race_counts = torch.zeros(len(race_labels)).to(device)

    # # Count occurrences of each attribute
    # for i in range(len(gender_labels)):
    #     gender_counts[i] = (max_g_indices == i).sum().float()

    # for i in range(len(race_labels)):
    #     race_counts[i] = (max_r_indices == i).sum().float()

    # # Calculate the distribution percentages
    # gender_distribution = gender_counts / probs_gender.shape[0]
    # race_distribution = race_counts / probs_race.shape[0]
    # return  gender_distribution, race_distribution
    return probs_gender

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Classifier("clip-vit-base-patch32", device)

    image_names, images = [], []
    for img in OUTPUTS.iterdir():
        image_names.append(str(img))
        images.append(load_image(img))

    text_classes = ["photo of a man", "photo of a woman" ]
    prompts = ["a photo of an Asian","a photo of a caucasian", "a photo of a black person", "a photo of a latin American"]
    outputs = model.forward(text_classes, images)
    outputs_race = model.forward(prompts, images)
    probs_gender = model.classify(outputs)
    probs_race = model.classify(outputs_race)
    # print(probs_gender)
    # print(probs_race)
    # print("image", text_classes)
    # for i, img in enumerate(image_names):
    #     print(img.split('/')[-1], probs[i])
    #     print(img.split('/')[-1], probs_race[i])
    gender_labels = ['Man', 'Woman']
    race_labels = ['asian', 'white', 'black', 'latino']

    # Initialize counters for each attribute
    gender_counts = {label: 0 for label in gender_labels}
    race_counts = {label: 0 for label in race_labels}

    # Iterate over each image's probabilities and count the occurrences of each attribute
    for g_prob, r_prob in zip(probs_gender, probs_race):
        # Get the index of the highest probability for gender and race
        max_g_index = np.argmax(g_prob)
        max_r_index = np.argmax(r_prob)

        # Increment the corresponding attribute count
        gender_counts[gender_labels[max_g_index]] += 1
        race_counts[race_labels[max_r_index]] += 1

    # Calculate the distribution percentages
    total_images = len(probs_gender)
    gender_distribution = {k: v / total_images for k, v in gender_counts.items()}
    race_distribution = {k: v / total_images for k, v in race_counts.items()}
    print(gender_distribution, race_distribution)