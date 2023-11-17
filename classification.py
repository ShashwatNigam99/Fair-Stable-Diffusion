from deepface import DeepFace
from pathlib import Path
import os, csv


def classify(img_path):
    objs = DeepFace.analyze(
        img_path = img_path, 
        actions = ['age', 'gender', 'race']
    )
    return objs

OUTPUTS = "./outputs/images"
OUTPUTS = Path(OUTPUTS).expanduser()

required_parameters = ['age', 'dominant_race', 'dominant_gender']

if __name__ == "__main__":
    os.environ.setdefault("DEEPFACE_HOME", "/home/hice1/mnigam9/scratch/cache")
    Path("/home/hice1/mnigam9/scratch/cache/.deepface/weights").mkdir(parents=True, exist_ok=True)
    
    output = []
    output.append(required_parameters+['image_path'])
    
    for img in OUTPUTS.iterdir():
        img_path = str(img)
        print(img_path)
        try:
            objs = classify(img_path)[0]
        except:
            print("Error in classification for image: ", img)
            continue
        
        line = []
        for param in required_parameters:
            line.append(objs[param])              
        line.append(img_path)        
        output.append(line)

    # Open the file in write mode
    with open(os.path.join(OUTPUTS,'output.csv'), 'w') as file:
        writer = csv.writer(file)
        # Write all rows at once
        writer.writerows(output)