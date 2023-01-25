# D<sup>3</sup>A: A Novel Denoising Diffusion Distortion Alignment Network for StyleGAN Inversion and Image Editing 

## Setup 
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```
Ninja Setup 
```
wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
sudo unzip ninja-linux.zip -d /usr/local/bin/
sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force
```
If you are using Pytorch Docker then no need to run Ninja Setup it is already installed in it.

For colab and local Ninja is requires.

## Prepare Images
We put some images from CelebA-HQ and Standford Cars in `./imgs`.   
For customized images, it is encouraged to first pre-process (align & crop) them, and then edit with our model.

## Pretrained Models 
We provide pretrained models on Human faces and Cars dataet.

| Dataset | Checkpoint | Link |
| :--- | :----------| :---------- | 
| Human Faces | iteration_14000.pt | [Download](https://drive.google.com/file/d/1TgW9zDs9Zj0xl2Ed-_3IorcN1wmwWa6i/view?usp=share_link) |
| Cars | iteration_6000.pt | [Download](https://drive.google.com/file/d/1qhfCW03m0RZtgJ11qAMv3qaaOrMVtsPi/view?usp=share_link) |

Use gdown to install the checkpoint 
```
pip install gdown
```
For Human Faces
```
gdown --fuzzy "https://drive.google.com/file/d/1TgW9zDs9Zj0xl2Ed-_3IorcN1wmwWa6i/view?usp=share_link"
```
For Cars 
```
gdown --fuzzy "https://drive.google.com/file/d/1qhfCW03m0RZtgJ11qAMv3qaaOrMVtsPi/view?usp=share_link"
```


## Inference
Modify `inference.sh` according to the follwing instructions, and run:   
(Currently we only support GPU for inference.)

```
bash inference.sh
```
for human faces 

| Args | Description
| :--- | :----------
| --images_dir | the path of images.
| --n_sample | number of images that you want to infer.
| --edit_attribute | We provide options of 'inversion', 'age', 'smile', 'eyes', 'lip' and 'beard' in the script.
| --edit_degree | control the degree of editing (works for 'age' and 'smile').

For Cars

| Args | Description
| :--- | :----------
| --images_dir | the path of images.
| --n_sample | number of images that you want to infer.
| --edit_attribute | We provide options of 'inversion', 'Pose1', 'Pose2', 'cube', 'color' and 'Grass' in the script.

## Training 
### Preparation
1. Download datasets and modify the dataset path in `./configs/paths_config.py` accordingly.
2. Download some pretrained models and put them in `./pretrained`.

Modify `options/train_options.py` and `train.sh` and run 
```
bash train.sh
```

## Metrics 
We provide metrics (MSE,LPIPS,SSIM,ID) in `metrics` Directory
For MSE,LPIPS,SSIM
```
python scripts/calc_losses_on_images.py --output_path=/path/to/experiment/inference_results --gt_path=/path/to/test_images
```
For ID Loss 
Download the [CurricularFace](https://drive.google.com/file/d/1f4IwVa2-Bn9vWLwB-bUwm53U_MlvinAj/view?usp=sharing) and [MTCNN](https://drive.google.com/file/d/1tJ7ih-wbCO6zc3JhI_1ZGjmwXKKaPlja/view?usp=sharing)
```
python calc_id_loss_parallel.py --output_path=/path/to/experiment/inference_results --gt_path=/path/to/test_images
```


