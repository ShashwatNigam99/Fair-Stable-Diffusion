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

def classifier(output_path="/home/okara7/Desktop/Fair-Stable-Diffusion/outputs/images"):
    required_parameters = ['age', 'dominant_race', 'dominant_gender']
    os.environ.setdefault("DEEPFACE_HOME", "/home/okara7/Desktop/Fair-Stable-Diffusion")
    local_path = ""
    Path(f'{local_path}.deepface/weights').mkdir(parents=True, exist_ok=True)
    OUTPUTS = output_path
    OUTPUTS = Path(OUTPUTS).expanduser()
    output = []
    output.append(required_parameters+['image_path'])
    for img in OUTPUTS.iterdir():
        img_path = str(img)

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
    headers = output[0]
    rows = output[1:]
    # Identify the indices for the relevant columns
    gender_index = headers.index('dominant_gender')
    race_index = headers.index('dominant_race')
    # Initialize dictionaries to count occurrences
    gender_counts = {}
    race_counts = {}
    # Count occurrences for each attribute
    for row in rows:
        gender = row[gender_index]
        race = row[race_index]
        gender_counts[gender] = gender_counts.get(gender, 0) + 1
        race_counts[race] = race_counts.get(race, 0) + 1
    # Convert counts to percentages
    total = len(rows)
    gender_percentages = {gender: count / total for gender, count in gender_counts.items()}
    race_percentages = {race: count / total for race, count in race_counts.items()}
    return gender_percentages, race_percentages

if __name__ == "__main__":
    gender_percentages, race_percentages = classifier()
    print(gender_percentages, race_percentages)
# classify_result('/home/okara7/Desktop/Fair-Stable-Diffusion/outputs/images/output.csv')