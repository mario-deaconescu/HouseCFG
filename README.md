# Creating and Editing Indoor Plans Using Generative Models
## BSc Thesis by Sabin-Mario Deaconescu

---

# How to run experiments

## HouseDiffusion

The source code for HouseDiffusion can be found as a submodule fork in ```lib/house_diffusion```.
The original repository is available at [aminshabani/house_diffusion](https://github.com/aminshabani/house_diffusion).

### Method 1 (Recommended)

1. Clone the original repository.
2. Follow the instructions in the original repository.
HouseCFG experiments used the HouseDiffusion model provided in the original repository.
3. Copy the generated ```processed_rplan``` into the HouseCFG project directory.
4. To sample, run ```python -m lib.house_diffusion.scripts.image_sample --batch_size 1 --set_name eval --target_set 8 --model_path models/house_diffusion/model250000.pt --num_samples 1 --output_path outputs/house_diffusion```.

### Method 2 (HouseCFG Fork)
1. Instead of the default location for the dataset, place the dataset files in `data/rplan` (i.e., ```*.json``` and ```list.txt```).
2. For the sampling code to run, at least one iteration of training must be run first. This is because of how the original code is written.
3. During HouseCFG experiments, some dataset entries caused errors (a known issue of HouseDiffusion). A list of working entries is provided in `lib/house_diffusion/list.txt`.
4. To sample, run ```python -m lib.house_diffusion.scripts.image_sample --batch_size 1 --set_name eval --target_set 8 --model_path models/house_diffusion/model250000.pt --num_samples 1 --output_path outputs/house_diffusion```.

## HouseGAN++

Run ```python -m lib.houseganpp.test```

# How to run HouseCFG App

## Backend

```bash
python -m virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
```