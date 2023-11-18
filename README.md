# Fair Stable Diffusion

Usage:
```
    conda env create -f environment.yml
    conda activate fairdiff
    python pipeline.py
    # this will create ./outputs folder containing images/ and hspace/ folders
    python classification.py
    # this will classify all images generated in ./outputs/images/ and create a csv annotations file
```

## Baseline
https://github.com/ml-research/Fair-Diffusion

## Face Classification model
https://github.com/serengil/deepface (subject to change)

## Stable diffusion model
https://huggingface.co/runwayml/stable-diffusion-v1-5



## Fair Diffusion Baseline

in `fair_diffusion/`, run  `pipeline.py` and defining the occupation in the function `gen_occupation_face()`.


