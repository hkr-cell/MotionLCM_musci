import os
import sys
import logging
import datetime
import os.path as osp
import time
from typing import Generator

import numpy as np
from tqdm.auto import tqdm
from omegaconf import OmegaConf

import torch
import swanlab
import diffusers
import transformers
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from diffusers.optimization import get_scheduler

from mld.config import parse_args, instantiate_from_config
from mld.data.get_data import get_dataset
from mld.models.modeltype.mld import MLD
from mld.utils.utils import print_table, set_seed, move_batch_to_device
from mld.utils.temos_utils import lengths_to_mask

from mld.models.modeltype.mld import MLD_music
from music_data import load_data_aist,prepare_dataloader,MoDaSeq,load_test_data_aist,calculate_metrics,MoSeq,visualizeAndWrite,visualizeAndWrite_novis
from Bailando import SepVQVAER
from collections import OrderedDict
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def guidance_scale_embedding(w: torch.Tensor, embedding_dim: int = 512,
                             dtype: torch.dtype = torch.float32) -> torch.Tensor:
    assert len(w.shape) == 1
    w = w * 1000.0

    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
    emb = w.to(dtype)[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1))
    assert emb.shape == (w.shape[0], embedding_dim)
    return emb


def append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


def scalings_for_boundary_conditions(timestep: torch.Tensor, sigma_data: float = 0.5,
                                     timestep_scaling: float = 10.0) -> tuple:
    c_skip = sigma_data ** 2 / ((timestep * timestep_scaling) ** 2 + sigma_data ** 2)
    c_out = (timestep * timestep_scaling) / ((timestep * timestep_scaling) ** 2 + sigma_data ** 2) ** 0.5
    return c_skip, c_out


def predicted_origin(
        model_output: torch.Tensor,
        timesteps: torch.Tensor,
        sample: torch.Tensor,
        prediction_type: str,
        alphas: torch.Tensor,
        sigmas: torch.Tensor
) -> torch.Tensor:
    if prediction_type == "epsilon":
        sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(alphas, timesteps, sample.shape)
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == "v_prediction":
        sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(alphas, timesteps, sample.shape)
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(f"Prediction type {prediction_type} currently not supported.")

    return pred_x_0


def extract_into_tensor(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class DDIMSolver:
    def __init__(self, alpha_cumprods: np.ndarray, timesteps: int = 1000, ddim_timesteps: int = 50) -> None:
        # DDIM sampling parameters
        step_ratio = timesteps // ddim_timesteps
        self.ddim_timesteps = (np.arange(1, ddim_timesteps + 1) * step_ratio).round().astype(np.int64) - 1
        self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]
        self.ddim_alpha_cumprods_prev = np.asarray(
            [alpha_cumprods[0]] + alpha_cumprods[self.ddim_timesteps[:-1]].tolist()
        )
        # convert to torch tensors
        self.ddim_timesteps = torch.from_numpy(self.ddim_timesteps).long()
        self.ddim_alpha_cumprods = torch.from_numpy(self.ddim_alpha_cumprods)
        self.ddim_alpha_cumprods_prev = torch.from_numpy(self.ddim_alpha_cumprods_prev)

    def to(self, device: torch.device) -> "DDIMSolver":
        self.ddim_timesteps = self.ddim_timesteps.to(device)
        self.ddim_alpha_cumprods = self.ddim_alpha_cumprods.to(device)
        self.ddim_alpha_cumprods_prev = self.ddim_alpha_cumprods_prev.to(device)
        return self

    def ddim_step(self, pred_x0: torch.Tensor, pred_noise: torch.Tensor,
                  timestep_index: torch.Tensor) -> torch.Tensor:
        alpha_cumprod_prev = extract_into_tensor(self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape)
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev


@torch.no_grad()
def update_ema(target_params: Generator, source_params: Generator, rate: float = 0.99) -> None:
    for tgt, src in zip(target_params, source_params):
        tgt.detach().mul_(rate).add_(src, alpha=1 - rate)


def main():
    cfg = parse_args()
    data = cfg.data
    external_wav_rate = cfg.ds_rate // cfg.external_wav_rate if hasattr(cfg, 'external_wav_rate') else 1
    external_wav_rate = cfg.music_relative_rate if hasattr(cfg, 'music_relative_rate') else external_wav_rate

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
    device = torch.device('cuda:7') if torch.cuda.is_available() else torch.device('cpu')
    set_seed(cfg.TRAIN.SEED_VALUE)

    name_time_str = osp.join(cfg.NAME, datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"))
    #name_time_str = osp.join(cfg.NAME, 'step1')
    output_dir = osp.join(cfg.FOLDER, name_time_str)
    os.makedirs(output_dir, exist_ok=False)
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=False)

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

    logger.info(f'Training guidance scale range (w): [{cfg.TRAIN.w_min}, {cfg.TRAIN.w_max}]')
    logger.info(f'EMA rate (mu): {cfg.TRAIN.ema_decay}')
    logger.info(f'Skipping interval (k): {cfg.model.scheduler.params.num_train_timesteps / cfg.TRAIN.num_ddim_timesteps}')
    logger.info(f'Loss type (huber or l2): {cfg.TRAIN.loss_type}')




    state_dict_model = torch.load('/mnt/disk_1/yufei/experiments_music/aist++/LCM_cod15_64_nums256_hid256_171520_wmin5_wmax15_ddim50_lr2e-4_u256/checkpoints/checkpoint-12730.ckpt', map_location="cpu")["state_dict"]
    lcm_key = 'denoiser.time_embedding.cond_proj.weight'
    is_lcm = False
    if lcm_key in state_dict_model:
        is_lcm = True
        time_cond_proj_dim = state_dict_model[lcm_key].shape[1]
        cfg.model.denoiser_music.params.time_cond_proj_dim = time_cond_proj_dim
    logger.info(f'Is LCM: {is_lcm}')
    base_model = MLD_music(cfg)






    vqvae = SepVQVAER(cfg.structure)
    checkpoint = torch.load(cfg.vqvae_weight, map_location=device)
    state_dict_vae = checkpoint['model']
    # new_state_dict = OrderedDict()
    # for k, v in state_dict_vae.items():
    #     name = k[7:]  # 去掉"module."前缀
    #     new_state_dict[name] = v

    vqvae.load_state_dict(state_dict_vae)
    base_model.load_state_dict(state_dict_model)
    vqvae = vqvae.to(device)
    base_model = base_model.to(device)
    vqvae.eval()
    base_model.eval()


    times = []
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
            time_begin = time.time()
            z_out = base_model.allsplit_step(split='val', batch=music_seq)  # 1 405 438
            time_diff =time.time()
            z_up = z_out[:, :, :64].permute(0, 2, 1)
            z_down = z_out[:, :, 64:].permute(0, 2, 1)
            zs = ([z_up], [z_down])

            with torch.no_grad():
                pose_sample, _, _ = vqvae.decode(([z_up], [z_down]))
            if cfg.global_vel:
                global_vel = pose_sample[:, :, :3].clone()
                pose_sample[:, 0, :3] = 0
                for iii in range(1, pose_sample.size(1)):
                    pose_sample[:, iii, :3] = pose_sample[:, iii - 1, :3] + global_vel[:, iii - 1, :]
            time_de = time.time()
            time_use = time_de-time_begin
            times.append(time_use)

            if isinstance(zs, tuple):
                quants_out[dance_names[i_eval]] = tuple(zs[ii][0].cpu().data.numpy()[0] for ii in range(len(zs)))
            else:
                quants_out[dance_names[i_eval]] = zs[0].cpu().data.numpy()[0]

            results.append(pose_sample)
        ep_path = visualizeAndWrite_novis(results, cfg, output_dir, dance_names, epoch_i, quants_out)
        metrics = calculate_metrics(ep_path)

        fid_k = metrics['fid_k']
        fid_m = metrics['fid_m']
        del times[0]
        print(len(times))
        print(times)
        print(np.mean(times))
        return metrics

    print(validation(0))

if __name__ == "__main__":
    main()