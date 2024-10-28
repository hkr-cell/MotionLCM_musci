import os
import sys
import logging
import datetime
import os.path as osp
import numpy as np

from tqdm.auto import tqdm
from omegaconf import OmegaConf

import torch
import swanlab
import diffusers
import transformers
from torch.utils.tensorboard import SummaryWriter
from diffusers.optimization import get_scheduler
from collections import OrderedDict
from mld.config import parse_args
from mld.data.get_data import get_dataset
from mld.models.modeltype.mld import MLD_music
from mld.utils.utils import print_table, set_seed, move_batch_to_device
from music_data import load_data_aist,prepare_dataloader,MoDaSeq,load_test_data_aist,calculate_metrics,MoSeq,visualizeAndWrite
from mld.models.architectures.vqvae import SepVQVAE_one
from Bailando import SepVQVAER
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():

    cfg = parse_args()
    data = cfg.data

    music_data, dance_data, dance_names = load_test_data_aist(
        data.test_real, \
        move=cfg.move, \
        rotmat=cfg.rotmat, \
        external_wav=cfg.external_wav if hasattr(cfg, 'external_wav') else None, \
        external_wav_rate=cfg.external_wav_rate if hasattr(cfg, 'external_wav_rate') else 1, \
        music_normalize=cfg.music_normalize if hasattr(cfg, 'music_normalize') else False, \
        wav_padding=cfg.wav_padding * (cfg.ds_rate // cfg.music_relative_rate) if hasattr(cfg, 'wav_padding') else 0)
    test_loader = torch.utils.data.DataLoader(
        MoDaSeq(music_data, dance_data),
        batch_size=1,
        shuffle=False
    )
    output_dir = '/mnt/disk_1/yufei/experiments_music/test/diffusion'
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    set_seed(cfg.TRAIN.SEED_VALUE)

    if cfg.TRAIN.model_ema:
        os.makedirs(f"{output_dir}/checkpoints_ema", exist_ok=False)

    if cfg.vis == "tb":
        writer = SummaryWriter(output_dir)
    elif cfg.vis == "swanlab":
        writer = swanlab.init(project="MotionLCM",
                              experiment_name=os.path.normpath(output_dir).replace(os.path.sep, "-"),
                              suffix=None, config=dict(**cfg), logdir=output_dir)
    else:
        raise ValueError(f"Invalid vis method: {cfg.vis}")

    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(osp.join(output_dir, 'output.log'))
    handlers = [file_handler, stream_handler]
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        handlers=handlers)
    logger = logging.getLogger(__name__)

    OmegaConf.save(cfg, osp.join(output_dir, 'config.yaml'))

    transformers.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_info()

    model = MLD_music(cfg)
    state_dict_model = torch.load('/mnt/disk_1/yufei/experiments_music/aist++/code30_256R_ep200/ckpt/epoch_10.pt_bsz128_lr2e-4_ffsize3072_layers15_heads_16_hidim512/checkpoints/checkpoint-209040.ckpt')[
        "state_dict"]

    vqvae = SepVQVAER(cfg.structure)
    checkpoint = torch.load('/home/yufei/MotionLCM-dev/Bailando_main/experiments/code30_256R_ep200/ckpt/epoch_10.pt', map_location=device)
    state_dict_vae = checkpoint['model']
    # new_state_dict = OrderedDict()
    # for k, v in state_dict_vae.items():
    #     name = k[7:]  # 去掉"module."前缀
    #     new_state_dict[name] = v
    vqvae.load_state_dict(state_dict_vae)

    vqvae.eval()
    model.eval()
    vqvae.to(device)
    model.to(device)
    model.load_state_dict(state_dict_model)

    @torch.no_grad()
    def validation(epoch_i):


        results = []
        quants_out = {}
        for i_eval, batch_eval in enumerate(tqdm(test_loader, desc='Generating Dance Poses')):

            music_seq, pose_seq = batch_eval
            pose_seq = pose_seq.to(device)

            music_seq = music_seq.float()

            music_ds_rate = cfg.ds_rate if not hasattr(cfg, 'external_wav') else cfg.external_wav_rate
            music_ds_rate = cfg.music_ds_rate if hasattr(cfg, 'music_ds_rate') else music_ds_rate
            music_relative_rate = cfg.music_relative_rate if hasattr(cfg, 'music_relative_rate') else cfg.ds_rate

            music_seq = music_seq[:, :, :cfg.n_music // music_ds_rate].contiguous().float()
            b, t, c = music_seq.size()
            music_seq = music_seq.view(b, t // music_ds_rate, c * music_ds_rate)
            music_relative_rate = cfg.music_relative_rate if hasattr(cfg, 'music_relative_rate') else cfg.ds_rate

            music_seq = music_seq[:, cfg.ds_rate // music_relative_rate:]



            music_seq = music_seq.to(device)
            z_out = model.allsplit_step(split='val', batch=music_seq) # 1 405 438
            z_up = z_out[:,:,:256].permute(0,2,1)
            z_down = z_out[:,:,256:].permute(0,2,1)

            with torch.no_grad():
                pose_sample,_,_ = vqvae.decode(([z_up], [z_down]))
            if cfg.global_vel:
                global_vel = pose_sample[:, :, :3].clone()
                pose_sample[:, 0, :3] = 0
                for iii in range(1, pose_sample.size(1)):
                    pose_sample[:, iii, :3] = pose_sample[:, iii - 1, :3] + global_vel[:, iii - 1, :]
            results.append(pose_sample)
        ep_path = visualizeAndWrite(results, cfg, output_dir, dance_names, epoch_i, quants_out)
        metrics = calculate_metrics(ep_path)

        fid_k = metrics['fid_k']
        fid_m = metrics['fid_m']

        return metrics

    print(validation(0))




if __name__ == "__main__":
    main()
