# python ./scripts/inference_cars.py \
# --images_dir=/content/drive/MyDrive/cars_train/cars_test --n_sample=500  --edit_attribute='Grass' --edit_degree=2.5  \
# --save_dir=/content/drive/MyDrive/diffusion_based/HFGI/test_results/cars  /content/drive/MyDrive/diffusion_based/HFGI/experiment/cars_encode_4/checkpoints/iteration_6000.pt

#### faces ###########
# python ./scripts/inference.py \
# --images_dir=/content/drive/MyDrive/diffusion_based/celeb/CelebAMask-HQ/CelebA-HQ-img   --edit_attribute='eyes' --edit_degree=2.5  \
# --save_dir=/content/drive/MyDrive/diffusion_based/HFGI/test_new_model  /content/drive/MyDrive/diffusion_based/HFGI/experiment/ffhq_11/checkpoints/iteration_14000.pt
#/content/drive/MyDrive/HFGI/experiment/ffhq_2_continued/checkpoints/iteration_20000.pt
#### Out of Doamin ####
python ./scripts/inference.py \
--images_dir=/content/drive/MyDrive/diffusion_based/HFGI/pti/input   --edit_attribute='inversion' --edit_degree=2.5  \
--save_dir=/content/drive/MyDrive/diffusion_based/HFGI/pti/results/single_stage  /content/drive/MyDrive/diffusion_based/HFGI/experiment/ffhq_11/checkpoints/iteration_14000.pt

# python ./scripts/inference.py \
# --images_dir=./test_imgs  --n_sample=100 --edit_attribute='lip'  \
# --save_dir=./experiment/inference_results    ./checkpoint/ckpt.pt 

# python ./scripts/inference.py \
# --images_dir=./test_imgs  --n_sample=100 --edit_attribute='beard'  \
# --save_dir=./experiment/inference_results  ./checkpoint/ckpt.pt 

# python ./scripts/inference.py \
# --images_dir=./test_imgs  --n_sample=100 --edit_attribute='eyes'  \
# --save_dir=./experiment/inference_results  ./checkpoint/ckpt.pt 

# python ./scripts/inference.py \
# --images_dir=./test_imgs  --n_sample=100 --edit_attribute='smile' --edit_degree=1.0  \
# --save_dir=./experiment/inference_results  ./checkpoint/ckpt.pt 

# python ./scripts/inference.py \
# --images_dir=./test_imgs  --n_sample=100 --edit_attribute='age' --edit_degree=3  \
# --save_dir=./experiment/inference_results  ./checkpoint/ckpt.pt 
