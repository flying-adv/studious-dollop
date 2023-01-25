
#### HUMAN FACES ###########
python ./scripts/inference.py \
--images_dir=./imgs/faces   --edit_attribute='inversion' \
--save_dir=./results  ./iteration_14000.pt

# python ./scripts/inference.py \
# --images_dir=./imgs/faces  --n_sample=100 --edit_attribute='lip'  \
# --save_dir=./results ./iteration_14000.pt

# python ./scripts/inference.py \
# --images_dir=./imgs/faces  --n_sample=100 --edit_attribute='beard'  \
# --save_dir=./results  ./iteration_14000.pt

# python ./scripts/inference.py \
# --images_dir=./imgs/faces  --n_sample=100 --edit_attribute='eyes'  \
# --save_dir=./results  ./iteration_14000.pt

# python ./scripts/inference.py \
# --images_dir=./imgs/faces  --n_sample=100 --edit_attribute='smile' --edit_degree=1.0  \
# --save_dir=./results  ./iteration_14000.pt

# python ./scripts/inference.py \
# --images_dir=./imgs/faces  --n_sample=100 --edit_attribute='age' --edit_degree=3  \
# --save_dir=./results  ./iteration_14000.pt

#### CAR ######
# python ./scripts/inference_cars.py \
# --images_dir=./imgs/cars --n_sample=100  --edit_attribute='inversion' \
# --save_dir=./results  ./iteration_6000.pt

# python ./scripts/inference_cars.py \
# --images_dir=./imgs/cars --n_sample=100  --edit_attribute='Pose1' \
# --save_dir=./results  ./iteration_6000.pt

# python ./scripts/inference_cars.py \
# --images_dir=./imgs/cars --n_sample=100  --edit_attribute='Pose2' \
# --save_dir=./results  ./iteration_6000.pt

# python ./scripts/inference_cars.py \
# --images_dir=./imgs/cars --n_sample=100  --edit_attribute='cube' \
# --save_dir=./results  ./iteration_6000.pt

# python ./scripts/inference_cars.py \
# --images_dir=./imgs/cars --n_sample=100  --edit_attribute='color' \
# --save_dir=./results  ./iteration_6000.pt

# python ./scripts/inference_cars.py \
# --images_dir=./imgs/cars --n_sample=100  --edit_attribute='Grass' \
# --save_dir=./results  ./iteration_6000.pt