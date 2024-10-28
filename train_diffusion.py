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
    external_wav_rate = cfg.ds_rate // cfg.external_wav_rate if hasattr(cfg,'external_wav_rate') else 1
    external_wav_rate = cfg.music_relative_rate if hasattr(cfg,'music_relative_rate') else external_wav_rate
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

    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    set_seed(cfg.TRAIN.SEED_VALUE)

    name_time_str = osp.join(cfg.NAME, 'code30_128R_nums256_ep220/ckpt/epoch_10.pt_bsz128_lr2e-4_ffsize3072_layers15_heads_16_hidim512')

    #name_time_str = osp.join(cfg.NAME, datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"))
    output_dir = osp.join(cfg.FOLDER, name_time_str)
    os.makedirs(output_dir, exist_ok=False)
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=False)
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



    model = MLD_music(cfg)  # /mnt/disk_1/yufei/experiments_music/aist++/aist_realcat_bsz128_lr2e-4_ffsize2048_layers13_heads_8_hidim1024/checkpoints/checkpoint-241200.ckpt

    vqvae = SepVQVAER(cfg.structure)
    print(cfg.structure)
    checkpoint = torch.load(cfg.vqvae_weight,map_location=device)
    state_dict = checkpoint['model']
    new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:]  # 去掉"module."前缀
    #     new_state_dict[name] = v
    vqvae.load_state_dict(state_dict)
    # vqvae = SepVQVAE_one(cfg.model.structure)
    #
    #
    # logger.info(f"Loading pre-trained model: {cfg.TRAIN.PRETRAINED}")
    # checkpoint = torch.load(cfg.TRAIN.PRETRAINED)
    # vqvae.load_state_dict(checkpoint['model'])

    vqvae.eval()
    vqvae.to(device)
    model.to(device)




    logger.info("learning_rate: {}".format(cfg.TRAIN.learning_rate))
    optimizer = torch.optim.AdamW(
        [{"params":model.denoiser.parameters()},{"params":model.music_emb.parameters()}],
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

    # EMA
    model_ema = None
    if cfg.TRAIN.model_ema:
        alpha = 1.0 - cfg.TRAIN.model_ema_decay
        logger.info(f'EMA alpha: {alpha}')
        model_ema = torch.optim.swa_utils.AveragedModel(model, device, lambda p0, p1, _: (1 - alpha) * p0 + alpha * p1)

    # Train!
    logger.info("***** Running training *****")
    logging.info(f"  Num examples = {len(training_data.dataset)}")
    logging.info(f"  Num Epochs = {cfg.TRAIN.max_train_epochs}")
    logging.info(f"  Instantaneous batch size per device = {cfg.TRAIN.BATCH_SIZE}")
    logging.info(f"  Total optimization steps = {cfg.TRAIN.max_train_steps}")

    global_step = 0
    all_indic_up = []
    all_indic_down = []
    codebook_size = 512



    @torch.no_grad()
    def validation_vqvae(epoch_i):
        results = []
        random_id = 0  # np.random.randint(0, 1e4)
        quants = {}

        # the restored error
        tot_euclidean_error = 0
        tot_eval_nums = 0
        tot_body_length = 0
        euclidean_errors = []
        for i_eval, batch_eval in enumerate(tqdm(test_loader, desc='Generating Dance Poses')):
            # Prepare data
            # pose_seq_eval = map(lambda x: x.to(self.device), batch_eval)
            music, pose_seq_eval = batch_eval
            pose_seq_eval = pose_seq_eval.to(device)
            src_pos_eval = pose_seq_eval[:, :]  #
            global_shift = src_pos_eval[:, :, :3].clone()
            if cfg.rotmat:
                # trans = pose_seq[:, :, :3]
                src_pos_eval = src_pos_eval[:, :, 3:]
            elif cfg.global_vel:
                print('Using Global Velocity')
                pose_seq_eval[:, :-1, :3] = pose_seq_eval[:, 1:, :3] - pose_seq_eval[:, :-1, :3]
                pose_seq_eval[:, -1, :3] = pose_seq_eval[:, -2, :3]
            else:
                src_pos_eval[:, :, :3] = 0

            b, t, c = src_pos_eval.size()
            src_pos_eval = src_pos_eval.to(device)
            z = vqvae.encode(src_pos_eval)


            pose_seq_out,zsup,zsdown = vqvae.decode(z)
            zsup = zsup.cpu().numpy()
            zsdown = zsdown.cpu().numpy()
            zsup = np.squeeze(zsup)
            zsdown = np.squeeze(zsdown)
            all_indic_up.append(zsup)
            all_indic_down.append(zsdown)


            if cfg.global_vel:
                print('Using Global Velocity')
                global_vel = pose_seq_out[:, :, :3].clone()
                pose_seq_out[:, 0, :3] = 0
                for iii in range(1, pose_seq_out.size(1)):
                    pose_seq_out[:, iii, :3] = pose_seq_out[:, iii - 1, :3] + global_vel[:, iii - 1, :]
            else:
                pose_seq_out[:, :, :3] = global_shift

            if cfg.rotmat:
                pose_seq_out = torch.cat([global_shift, pose_seq_out], dim=2)
            results.append(pose_seq_out)

            indexs = np.arange(t)

        ep_path = visualizeAndWrite(results, cfg, output_dir, dance_names, epoch_i, quants)
        print(ep_path)
        metrics = calculate_metrics(ep_path)
        print(metrics)




    @torch.no_grad()
    def validation(epoch_i):
        model.denoiser.eval()
        model.music_emb.eval()


        results = []
        quants_out = {}
        for i_eval, batch_eval in enumerate(tqdm(test_loader, desc='Generating Dance Poses')):

            music_seq, pose_seq = batch_eval
            #music_seq = music_seq.to(device)
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
            z_out = model.allsplit_step(split='val', batch=music_seq) # 1 405 438
            z_up = z_out[:,:,:128].permute(0,2,1)
            z_down = z_out[:,:,128:].permute(0,2,1)

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


        model.denoiser.train()
        model.music_emb.train()
        print(metrics)
        return fid_k, fid_m

    # validation_vqvae(0)
    #
    # combined_indices_up = np.concatenate(all_indic_up)
    # combined_indices_down = np.concatenate(all_indic_down)
    #
    # unique_codes_up = np.unique(combined_indices_up)
    # num_unique_codes_up = len(unique_codes_up)
    # unique_codes_down = np.unique(combined_indices_down)
    # num_unique_codes_down = len(unique_codes_down)
    #
    # codebook_utilization_up = num_unique_codes_up / codebook_size
    #
    # print(f"UP Codebook utilization across the dataset: {codebook_utilization_up * 100:.2f}%")
    #
    # codebook_utilization_down = num_unique_codes_down / codebook_size
    #
    # print(f"DOWN Codebook utilization across the dataset: {codebook_utilization_down * 100:.2f}%")
    #
    # exit(0)
    min_fidk, min_fidm = validation(0)


    if cfg.TRAIN.model_ema:
        validation(model_ema, ema=True)

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

            with torch.no_grad():
                z = vqvae.encode(pose_seq)



            z[0][0] = z[0][0].permute(0,2,1) # bs 30 512
            z[1][0] = z[1][0].permute(0,2,1)
            input = torch.cat([z[0][0],z[1][0]],-1)

            loss_dict = model.allsplit_step('train', [input,music_seq[:, cfg.ds_rate//music_relative_rate:]])

            loss = loss_dict['diff_loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.denoiser.parameters(), cfg.TRAIN.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(model.music_emb.parameters(), cfg.TRAIN.max_grad_norm)


            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            progress_bar.update(1)
            global_step += 1


            if model_ema and global_step % cfg.TRAIN.model_ema_steps == 0:
                model_ema.update_parameters(model)

            if global_step % cfg.TRAIN.checkpointing_steps == 0:
                save_path = os.path.join(output_dir, 'checkpoints', f"checkpoint-{global_step}.ckpt")
                ckpt = dict(state_dict=model.state_dict())
                model.on_save_checkpoint(ckpt)
                torch.save(ckpt, save_path)
                logger.info(f"Saved state to {save_path}")

                if cfg.TRAIN.model_ema:
                    save_path = os.path.join(output_dir, 'checkpoints_ema', f"checkpoint-{global_step}.ckpt")
                    ckpt = dict(state_dict=model_ema.state_dict())
                    model_ema.on_save_checkpoint(ckpt)
                    torch.save(ckpt, save_path)
                    logger.info(f"Saved EMA state to {save_path}")

            if global_step % cfg.TRAIN.validation_steps == 0:
                fidk, fidm = validation(global_step)
                if cfg.TRAIN.model_ema:
                    validation(model_ema, ema=True)

                if fidk < min_fidk:
                    min_fidk = fidk
                    save_path = os.path.join(output_dir, 'checkpoints',
                                             f"checkpoint-{global_step}-fidk-{round(fidk, 3)}.ckpt")
                    ckpt = dict(state_dict=model.state_dict())
                    model.on_save_checkpoint(ckpt)
                    torch.save(ckpt, save_path)
                    logger.info(f"Saved state to {save_path} with fidk:{round(fidk, 3)}")

                if fidm < min_fidm:
                    min_fidm = fidm
                    save_path = os.path.join(output_dir, 'checkpoints',
                                             f"checkpoint-{global_step}-fidm-{round(fidm, 3)}.ckpt")
                    ckpt = dict(state_dict=model.state_dict())
                    model.on_save_checkpoint(ckpt)
                    torch.save(ckpt, save_path)
                    logger.info(f"Saved state to {save_path} with fidm:{round(fidm, 3)}")

            logs = {"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            for k, v in logs.items():
                if cfg.vis == "tb":
                    writer.add_scalar(f"Train/{k}", v, global_step=global_step)
                elif cfg.vis == "swanlab":
                    writer.log({f"Train/{k}": v}, step=global_step)

            if global_step >= cfg.TRAIN.max_train_steps:
                save_path = os.path.join(output_dir, 'checkpoints', f"checkpoint-last.ckpt")
                ckpt = dict(state_dict=model.state_dict())
                model.on_save_checkpoint(ckpt)
                torch.save(ckpt, save_path)

                if cfg.TRAIN.model_ema:
                    save_path = os.path.join(output_dir, 'checkpoints_ema', f"checkpoint-last.ckpt")
                    ckpt = dict(state_dict=model_ema.state_dict())
                    model_ema.on_save_checkpoint(ckpt)
                    torch.save(ckpt, save_path)

                exit(0)


if __name__ == "__main__":
    main()
