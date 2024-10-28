# MotionLCM_music
## Setup Environment
Our method is trained on python3.9 torch1.13.1,install the packages in ***requirments.txt***
## Data
We use AIST++ to train our model,you can check [Bailando](https://github.com/lisiyao21/Bailando/) to download the processed data or process the data by yourself.
Change your data path in ***vqvae.yaml,vqvaer.yaml,diffusion.yaml,motionlcm_t2m.yaml***
## Pretrained Model
You can download our pretrained model in [there](https://drive.google.com/drive/folders/1DSqEPUpxGRkHavLwYknPFnzGtja97qWY?usp=sharing),add the folder checkpoint to your code,it should cover ***vqvaer,diffusion,lcm***
Change your model path in ***vqvaer.yaml,diffusion.yaml,motionlcm_t2m.yaml***
## Train Your Model
If you want to train model by yourself,follow
### Train VQVAE
> python train_vqvae.py --cfg configs/modules/vqvae.yaml
> python train_vqvae.py --cfg configs/modules/vqvaer.yaml #Don't forget set the vqvae_weight
### Train Diffusion
> python train_diffusion.py --cfg configs/diffusion.yaml  #Don't forget set the vqvae_weight
### Train Motion_LCM
> python train_motionlcm.py --cfg configs/motionlcm.yaml #Don't forget set the vqvae_weight and PRETRAINED diffusion
## Test your model
