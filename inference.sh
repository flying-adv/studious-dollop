
#### HUMAN FACES ###########
# python ./scripts/inference.py \
# --images_dir=./CelebA-HQ-img   --edit_attribute='inversion' \
# --save_dir=./results  ./pretrained_models/iteration_14000.pt

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

#### CAR ######
# python ./scripts/inference_cars.py \
# --images_dir=./cars_train/cars_test --n_sample=500  --edit_attribute='Grass' \
# --save_dir=./test_results/cars  ./pretrained_models/iteration_6000.pt