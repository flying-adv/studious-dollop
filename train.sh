python ./scripts/train.py   --dataset_type='ffhq_encode'  --start_from_latent_avg \
--id_lambda=1.1  --val_interval=200000 --save_interval=2000 --max_steps=100000  --stylegan_size=1024 --is_train=True \
--distortion_scale=0.15 --aug_rate=0.9 --res_lambda=0 --lpips_lambda=3.2 \
--stylegan_weights='./pretrained/stylegan2-ffhq-config-f.pt' --checkpoint_path='/content/drive/MyDrive/diffusion_based/HFGI/experiment/ffhq_10/checkpoints/iteration_6000.pt'  \
--workers=48  --batch_size=4 --test_batch_size=1 --test_workers=48 --exp_dir='./experiment/ffhq_11'
