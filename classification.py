from pathlib import Path
from torchvision import transforms, models
import torch
from PIL import Image


GENDER_MODEL_PATH = '/home/hice1/mnigam9/scratch/cache/face_classifiers/face_gender_classification_transfer_learning_with_ResNet18.pth'
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def build_model(model_path):
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path))
    return model

def get_classification(model, image):
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
    return preds

def load_image(image_path):
    return image_transforms(Image.open(image_path))


OUTPUTS = "/home/hice1/mnigam9/scratch/cache/deltahspace"
OUTPUTS = Path(OUTPUTS).expanduser()

if __name__ == "__main__":
    model = build_model(GENDER_MODEL_PATH)
    model = model.cuda()
    for img in OUTPUTS.iterdir():
        image = load_image(str(img)).unsqueeze(0).cuda()

        classification = get_classification(model, image)
        
        print(str(img), classification)
