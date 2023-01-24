python ./scripts/train.py   --dataset_type='cars_encode'  --start_from_latent_avg \
--id_lambda=0.8  --val_interval=200000 --save_interval=2000 --max_steps=100000  --stylegan_size=512 --is_train=True \
--distortion_scale=0.15 --aug_rate=0.9 --res_lambda=0 --lpips_lambda=2.8 \
--workers=48  --batch_size=8 --test_batch_size=1 --test_workers=48 --exp_dir='./experiment/cars_encode'
