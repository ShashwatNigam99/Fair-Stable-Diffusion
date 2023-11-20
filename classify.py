from deepface import DeepFace
from pathlib import Path
import os, csv
import pandas as pd

def classify(img_path):
    objs = DeepFace.analyze(
        img_path = img_path,
        actions = ['age', 'gender', 'race']
    )
    return objs

def classifier():
    required_parameters = ['age', 'dominant_race', 'dominant_gender']
    os.environ.setdefault("DEEPFACE_HOME", "/home/okara7/Desktop/Fair-Stable-Diffusion")
    local_path = ""
    Path(f'{local_path}.deepface/weights').mkdir(parents=True, exist_ok=True)
    OUTPUTS = "./outputs/images"
    OUTPUTS = Path(OUTPUTS).expanduser()
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
        
def classify_result(file_path):
    data = pd.read_csv(file_path)
    # Calculate percentage distribution for 'dominant_gender'
    gender_distribution = data['dominant_gender'].value_counts(normalize=True).to_dict()
    # Calculate percentage distribution for 'dominant_race'
    race_distribution = data['dominant_race'].value_counts(normalize=True).to_dict()
    # Convert to percentage format (optional)
    gender_distribution_percent = {k: v * 100 for k, v in gender_distribution.items()}
    race_distribution_percent = {k: v * 100 for k, v in race_distribution.items()}
    print("Gender Distribution (in %):", gender_distribution_percent)
    print("Race Distribution (in %):", race_distribution_percent)
    return gender_distribution_percent, race_distribution_percent

classifier()
classify_result('/home/okara7/Desktop/Fair-Stable-Diffusion/outputs/images/output.csv')