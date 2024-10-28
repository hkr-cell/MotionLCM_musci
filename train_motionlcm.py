import os
import sys
import logging
import datetime
import os.path as osp
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
from music_data import load_data_aist,prepare_dataloader,MoDaSeq,load_test_data_aist,calculate_metrics,MoSeq,visualizeAndWrite
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
    train_music_data, train_dance_data, _ = load_data_aist(
        data.train_real, interval=data.seq_len, move=cfg.move if hasattr(cfg, 'move') else 64,
        rotmat=cfg.rotmat, \
        external_wav=cfg.external_wav if hasattr(cfg, 'external_wav') else None, \
        external_wav_rate=external_wav_rate, \
        music_normalize=cfg.music_normalize if hasattr(cfg, 'music_normalize') else False, \
        wav_padding=cfg.wav_padding * (cfg.ds_rate // cfg.music_relative_rate) if hasattr(cfg, 'wav_padding') else 0)
    training_data = prepare_dataloader(train_music_data, train_dance_data, cfg.batch_size)

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
    device = torch.device('cuda:3') if torch.cuda.is_available() else torch.device('cpu')
    set_seed(cfg.TRAIN.SEED_VALUE)

    name_time_str = osp.join(cfg.NAME, 'LCM_cod15_64_nums256_hid256_171520_wmin5_wmax15_ddim50_lr2e-4_u768')
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



    logger.info(f"Loading pretrained model: {cfg.TRAIN.PRETRAINED}")
    state_dict_model = torch.load(cfg.TRAIN.PRETRAINED, map_location="cpu")["state_dict"]
    base_model = MLD_music(cfg)
    base_model.load_state_dict(state_dict_model)

    scheduler = base_model.scheduler
    alpha_schedule = torch.sqrt(scheduler.alphas_cumprod)
    sigma_schedule = torch.sqrt(1 - scheduler.alphas_cumprod)
    solver = DDIMSolver(
        scheduler.alphas_cumprod.numpy(),
        timesteps=scheduler.config.num_train_timesteps,
        ddim_timesteps=cfg.TRAIN.num_ddim_timesteps)

    base_model.to(device)

    vqvae = SepVQVAER(cfg.structure)
    checkpoint = torch.load(cfg.vqvae_weight, map_location=device)
    state_dict_vae = checkpoint['model']

    vqvae.load_state_dict(state_dict_vae)
    text_encoder = base_model.music_emb
    teacher_unet = base_model.denoiser
    vqvae = vqvae.to(device)
    vqvae.eval()
    vqvae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    teacher_unet.requires_grad_(False)

    # Apply CFG here (Important!!!)
    cfg.model.denoiser_music.params.time_cond_proj_dim = cfg.TRAIN.unet_time_cond_proj_dim
    unet = instantiate_from_config(cfg.model.denoiser_music)
    unet.load_state_dict(teacher_unet.state_dict(), strict=False)
    target_unet = instantiate_from_config(cfg.model.denoiser_music)
    target_unet.load_state_dict(teacher_unet.state_dict(), strict=False)

    # Only evaluate the online network
    base_model.denoiser = unet

    unet = unet.to(device)
    target_unet = target_unet.to(device)
    target_unet.requires_grad_(False)

    # Also move the alpha and sigma noise schedules to device
    alpha_schedule = alpha_schedule.to(device)
    sigma_schedule = sigma_schedule.to(device)
    solver = solver.to(device)

    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=cfg.TRAIN.learning_rate,
        betas=(cfg.TRAIN.adam_beta1, cfg.TRAIN.adam_beta2),
        weight_decay=cfg.TRAIN.adam_weight_decay,
        eps=cfg.TRAIN.adam_epsilon)

    if cfg.TRAIN.max_train_steps == -1:
        assert cfg.TRAIN.max_train_epochs != -1
        cfg.TRAIN.max_train_steps = cfg.TRAIN.max_train_epochs * len(training_data)

    if cfg.TRAIN.checkpointing_steps == -1:
        assert cfg.TRAIN.checkpointing_epochs != -1
        cfg.TRAIN.checkpointing_steps = cfg.TRAIN.checkpointing_epochs * len(training_data)

    if cfg.TRAIN.validation_steps == -1:
        assert cfg.TRAIN.validation_epochs != -1
        cfg.TRAIN.validation_steps = cfg.TRAIN.validation_epochs * len(training_data)

    lr_scheduler = get_scheduler(
        cfg.TRAIN.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.TRAIN.lr_warmup_steps,
        num_training_steps=cfg.TRAIN.max_train_steps)

    uncond_prompt = torch.zeros([cfg.batch_size, 29, 438], dtype=torch.float).to(device)
    uncond_prompt_embeds = text_encoder(uncond_prompt)

    # Train!
    logger.info("***** Running training *****")
    logging.info(f"  Num examples = {len(training_data.dataset)}")
    logging.info(f"  Num Epochs = {cfg.TRAIN.max_train_epochs}")
    logging.info(f"  Instantaneous batch size per device = {cfg.TRAIN.BATCH_SIZE}")
    logging.info(f"  Total optimization steps = {cfg.TRAIN.max_train_steps}")

    global_step = 0

    @torch.no_grad()
    def validation(epoch_i):
        base_model.eval()
        results = []
        quants_out = {}
        for i_eval, batch_eval in enumerate(tqdm(test_loader, desc='Generating Dance Poses')):

            music_seq, pose_seq = batch_eval
            # music_seq = music_seq.to(device)
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
            # it is just music_seq[:, 1:], ignoring the first music feature

            music_seq = music_seq.to(device)
            z_out = base_model.allsplit_step(split='val', batch=music_seq)  # 1 405 438
            z_up = z_out[:, :, :64].permute(0, 2, 1)
            z_down = z_out[:, :, 64:].permute(0, 2, 1)

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

        for mk, mv in metrics.items():
            if cfg.vis == "tb":
                writer.add_scalar(mk, mv, global_step=global_step)
            elif cfg.vis == "swanlab":
                writer.log({mk: mv}, step=global_step)
        base_model.train()
        return fid_k, fid_m

    min_fidk, min_fidm = validation(0)

    progress_bar = tqdm(range(0, cfg.TRAIN.max_train_steps), desc="Steps")
    while True:
        for step, batch in enumerate(training_data):
            music_seq, pose_seq = batch
            music_seq = music_seq.to(device)
            pose_seq = pose_seq.to(device)
            music_seq = music_seq.float()
            pose_seq[:, :, :3] = 0
            # print(pose_seq.size()) 32 240 72
            # print(music_seq.size())32 30  438

            music_ds_rate = cfg.ds_rate if not hasattr(cfg, 'external_wav') else cfg.external_wav_rate
            music_ds_rate = cfg.music_ds_rate if hasattr(cfg, 'music_ds_rate') else music_ds_rate
            music_relative_rate = cfg.music_relative_rate if hasattr(cfg, 'music_relative_rate') else cfg.ds_rate
            music_seq = music_seq[:, :, :cfg.n_music // music_ds_rate].contiguous().float()

            b, t, c = music_seq.size()
            music_seq = music_seq.view(b, t // music_ds_rate, c * music_ds_rate)
            # print(music_seq.size())  32 30 438

            if hasattr(cfg, 'music_normalize') and cfg.music_normalize:
                print('Normalize!')
                music_seq = music_seq / (t // music_ds_rate * 1.0)

            # Encode motions to latents
            with torch.no_grad():
                z = vqvae.encode(pose_seq)
            z[0][0] = z[0][0].permute(0, 2, 1)  # bs 30 512
            z[1][0] = z[1][0].permute(0, 2, 1)
            input = torch.cat([z[0][0], z[1][0]], -1)

            music_input = music_seq[:, cfg.ds_rate // music_relative_rate:]

            prompt_embeds = text_encoder(music_input)

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(input)
            bsz = input.shape[0]

            # Sample a random timestep for each image t_n ~ U[0, N - k - 1] without bias.
            topk = scheduler.config.num_train_timesteps // cfg.TRAIN.num_ddim_timesteps
            index = torch.randint(0, cfg.TRAIN.num_ddim_timesteps, (bsz,), device=input.device).long()
            start_timesteps = solver.ddim_timesteps[index]
            timesteps = start_timesteps - topk
            timesteps = torch.where(timesteps < 0, torch.zeros_like(timesteps), timesteps)

            # Get boundary scalings for start_timesteps and (end) timesteps.
            c_skip_start, c_out_start = scalings_for_boundary_conditions(start_timesteps)
            c_skip_start, c_out_start = [append_dims(x, input.ndim) for x in [c_skip_start, c_out_start]]
            c_skip, c_out = scalings_for_boundary_conditions(timesteps)
            c_skip, c_out = [append_dims(x, input.ndim) for x in [c_skip, c_out]]

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process) [z_{t_{n + k}} in Algorithm 1]
            noisy_model_input = scheduler.add_noise(input, noise, start_timesteps)

            # Sample a random guidance scale w from U[w_min, w_max] and embed it
            w = (cfg.TRAIN.w_max - cfg.TRAIN.w_min) * torch.rand((bsz,)) + cfg.TRAIN.w_min
            w_embedding = guidance_scale_embedding(w, embedding_dim=cfg.TRAIN.unet_time_cond_proj_dim)
            w = append_dims(w, input.ndim)
            # Move to U-Net device and dtype
            w = w.to(device=input.device, dtype=input.dtype)
            w_embedding = w_embedding.to(device=input.device, dtype=input.dtype)

            # Get online LCM prediction on z_{t_{n + k}}, w, c, t_{n + k}
            noise_pred = unet(
                noisy_model_input,
                start_timesteps,
                timestep_cond=w_embedding,
                encoder_hidden_states=prompt_embeds)

            pred_x_0 = predicted_origin(
                noise_pred,
                start_timesteps,
                noisy_model_input,
                scheduler.config.prediction_type,
                alpha_schedule,
                sigma_schedule)

            model_pred = c_skip_start * noisy_model_input + c_out_start * pred_x_0

            # Use the ODE solver to predict the k-th step in the augmented PF-ODE trajectory after
            # noisy_latents with both the conditioning embedding c and unconditional embedding 0
            # Get teacher model prediction on noisy_latents and conditional embedding
            with torch.no_grad():
                cond_teacher_output = teacher_unet(
                    noisy_model_input,
                    start_timesteps,
                    encoder_hidden_states=prompt_embeds)
                cond_pred_x0 = predicted_origin(
                    cond_teacher_output,
                    start_timesteps,
                    noisy_model_input,
                    scheduler.config.prediction_type,
                    alpha_schedule,
                    sigma_schedule)

                # Get teacher model prediction on noisy_latents and unconditional embedding
                uncond_teacher_output = teacher_unet(
                    noisy_model_input,
                    start_timesteps,
                    encoder_hidden_states=uncond_prompt_embeds[:bsz])
                uncond_pred_x0 = predicted_origin(
                    uncond_teacher_output,
                    start_timesteps,
                    noisy_model_input,
                    scheduler.config.prediction_type,
                    alpha_schedule,
                    sigma_schedule)

                # Perform "CFG" to get z_prev estimate (using the LCM paper's CFG formulation)
                pred_x0 = cond_pred_x0 + w * (cond_pred_x0 - uncond_pred_x0)
                pred_noise = cond_teacher_output + w * (cond_teacher_output - uncond_teacher_output)
                x_prev = solver.ddim_step(pred_x0, pred_noise, index)

            # Get target LCM prediction on z_prev, w, c, t_n
            with torch.no_grad():
                target_noise_pred = target_unet(
                    x_prev.float(),
                    timesteps,
                    timestep_cond=w_embedding,
                    encoder_hidden_states=prompt_embeds)
                pred_x_0 = predicted_origin(
                    target_noise_pred,
                    timesteps,
                    x_prev,
                    scheduler.config.prediction_type,
                    alpha_schedule,
                    sigma_schedule)
                target = c_skip * x_prev + c_out * pred_x_0

            # Calculate loss
            if cfg.TRAIN.loss_type == "l2":
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            elif cfg.TRAIN.loss_type == "huber":
                loss = torch.mean(
                    torch.sqrt(
                        (model_pred.float() - target.float()) ** 2 + cfg.TRAIN.huber_c ** 2) - cfg.TRAIN.huber_c
                )
            else:
                raise ValueError(f'Unknown loss type: {cfg.TRAIN.loss_type}.')

            # Back propagate on the online student model (`unet`)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), cfg.TRAIN.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            # Make EMA update to target student model parameters
            update_ema(target_unet.parameters(), unet.parameters(), cfg.TRAIN.ema_decay)
            progress_bar.update(1)
            global_step += 1

            if global_step % cfg.TRAIN.checkpointing_steps == 0:
                save_path = os.path.join(output_dir, 'checkpoints', f"checkpoint-{global_step}.ckpt")
                ckpt = dict(state_dict=base_model.state_dict())
                base_model.on_save_checkpoint(ckpt)
                torch.save(ckpt, save_path)
                logger.info(f"Saved state to {save_path}")

            if global_step % cfg.TRAIN.validation_steps == 0:
                cur_fidk, cur_fidm = validation(global_step)
                if cur_fidk < min_fidk:
                    min_fidk = cur_fidk
                    save_path = os.path.join(output_dir, 'checkpoints',
                                             f"checkpoint-{global_step}-fidk-{round(min_fidk, 3)}.ckpt")
                    ckpt = dict(state_dict=base_model.state_dict())
                    base_model.on_save_checkpoint(ckpt)
                    torch.save(ckpt, save_path)
                    logger.info(f"Saved state to {save_path} with fidk:{round(min_fidk, 3)}")

                if cur_fidm < min_fidm:
                    min_fidm = cur_fidm
                    save_path = os.path.join(output_dir, 'checkpoints',
                                             f"checkpoint-{global_step}-fidm-{round(min_fidm, 3)}.ckpt")
                    ckpt = dict(state_dict=base_model.state_dict())
                    base_model.on_save_checkpoint(ckpt)
                    torch.save(ckpt, save_path)
                    logger.info(f"Saved state to {save_path} with fidm:{round(min_fidm, 3)}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            if cfg.vis == "tb":
                writer.add_scalar('loss', logs['loss'], global_step=global_step)
                writer.add_scalar('lr', logs['lr'], global_step=global_step)
            elif cfg.vis == "swanlab":
                writer.log({'loss': logs['loss'], 'lr': logs['lr']}, step=global_step)

            if global_step >= cfg.TRAIN.max_train_steps:
                save_path = os.path.join(output_dir, 'checkpoints', f"checkpoint-last.ckpt")
                ckpt = dict(state_dict=base_model.state_dict())
                base_model.on_save_checkpoint(ckpt)
                torch.save(ckpt, save_path)
                exit(0)


if __name__ == "__main__":
    main()