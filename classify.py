from deepface import Deepface
import os
import json
#image path
img_path = "./outputs/images/"
attributes = ['age', 'gender', 'race']
png_files = []
demography_json = []
for filename in os.listdir(img_path):
    if filename.endswith(".png"):
        file_path = os.path.join(img_path, filename)
        png_files.append(file_path)

for i in range(len(png_files)):
    demography = Deepface.analyze(png_files[i], attributes)
    demography_json.append(json.load(demography))

print(demography_json)