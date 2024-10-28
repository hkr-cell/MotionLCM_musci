import os
import datetime
import os.path as osp
import random
from tqdm.auto import tqdm


import torch
from mld.config import parse_args

from mld.utils.utils import print_table, set_seed, move_batch_to_device
from mld.data_process.music_data import load_data_aist,prepare_dataloader,MoDaSeq,load_test_data_aist,calculate_metrics,MoSeq,visualizeAndWrite
from mld.models.architectures.vqvae import SepVQVAER,SepVQVAE
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



    name_time_str = osp.join(cfg.NAME, datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"))
    output_dir = osp.join(cfg.FOLDER, name_time_str)
    os.makedirs(output_dir, exist_ok=False)
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=False)
    if cfg.structure.name == 'SepVQVAER':
        model = SepVQVAE(cfg.structure)
    else if cfg.structure.name == 'SepVQVAE':
        model = SepVQVAE(cfg.structure)

    self.optimizer = optim(itertools.chain(self.model.parameters(),
                                           ),
                           **config.kwargs)
    self.schedular = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, **config.schedular_kwargs)
    updates = 0



    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    # if args.cuda:
    torch.cuda.manual_seed(cfg.seed)


    # Training Loop
    for epoch_i in range(1, cfg.epoch + 1):
        log.set_progress(epoch_i, len(training_data))

        for batch_i, batch in enumerate(training_data):
            # LR Scheduler missing
            # pose_seq = map(lambda x: x.to(self.device), batch)
            trans = None
            pose_seq = batch.to(self.device)
            if config.rotmat:
                trans = pose_seq[:, :, :3]
                pose_seq = pose_seq[:, :, 3:]
            elif config.global_vel:
                # print('Use vel!')
                # print(pose_seq[:, : :3])
                pose_seq[:, :-1, :3] = pose_seq[:, 1:, :3] - pose_seq[:, :-1, :3]
                pose_seq[:, -1, :3] = pose_seq[:, -2, :3]
                pose_seq = pose_seq.clone().detach()
            else:
                pose_seq[:, :, :3] = 0
            # print(pose_seq.size())
            optimizer.zero_grad()

            output, loss, metrics = model(pose_seq)

            loss = loss.mean()

            loss.backward()

            # update parameters
            optimizer.step()

            stats = {
                'updates': updates,
                'loss': loss.item(),
                # 'velocity_loss_if_have': metrics[0]['velocity_loss'].item() + metrics[1]['velocity_loss'].item(),
                # 'acc_loss_if_have': metrics[0]['acceleration_loss'].item() + metrics[1]['acceleration_loss'].item()
            }
            # if epoch_i % self.config.log_per_updates == 0:
            updates += 1

        checkpoint = {
            'model': model.state_dict(),
            'config': config,
            'epoch': epoch_i
        }

        # # Save checkpoint
        if epoch_i % config.save_per_epochs == 0 or epoch_i == 1:
            filename = os.path.join(self.ckptdir, f'epoch_{epoch_i}.pt')
            torch.save(checkpoint, filename)
        # Eval
        if epoch_i % config.test_freq == 0:
            with torch.no_grad():
                print("Evaluation...")
                model.eval()
                results = []
                random_id = 0  # np.random.randint(0, 1e4)
                quants = {}
                for i_eval, batch_eval in enumerate(tqdm(test_loader, desc='Generating Dance Poses')):
                    # Prepare data
                    # pose_seq_eval = map(lambda x: x.to(self.device), batch_eval)
                    pose_seq_eval = batch_eval.to(self.device)
                    src_pos_eval = pose_seq_eval[:, :]  #
                    global_shift = src_pos_eval[:, :, :3].clone()
                    if config.rotmat:
                        # trans = pose_seq[:, :, :3]
                        src_pos_eval = src_pos_eval[:, :, 3:]
                    elif config.global_vel:
                        src_pos_eval[:, :-1, :3] = src_pos_eval[:, 1:, :3] - src_pos_eval[:, :-1, :3]
                        src_pos_eval[:, -1, :3] = src_pos_eval[:, -2, :3]
                    else:
                        src_pos_eval[:, :, :3] = 0

                    pose_seq_out, loss, _ = model(src_pos_eval)  # first 20 secs

                    if config.rotmat:
                        pose_seq_out = torch.cat([global_shift, pose_seq_out], dim=2)
                    if config.global_vel:
                        global_vel = pose_seq_out[:, :, :3].clone()
                        pose_seq_out[:, 0, :3] = 0
                        for iii in range(1, pose_seq_out.size(1)):
                            pose_seq_out[:, iii, :3] = pose_seq_out[:, iii - 1, :3] + global_vel[:, iii - 1, :]
                        # print('Use vel!')
                        # print(pose_seq_out[:, :, :3])
                    else:
                        _, t, _ = pose_seq_out.size()
                        pose_seq_out[:, :, :3] = global_shift[:, :t, :]
                    results.append(pose_seq_out)
                    if not True:  # config.structure.use_bottleneck:
                        quants_pred = model.encode(src_pos_eval)
                        if isinstance(quants_pred, tuple):
                            quants[self.dance_names[i_eval]] = tuple(
                                quants_pred[ii][0].cpu().data.numpy()[0] for ii in range(len(quants_pred)))
                        else:
                            quants[self.dance_names[i_eval]] = model.encode(src_pos_eval)[0].cpu().data.numpy()[0]
                    else:
                        quants = None
                visualizeAndWrite(results, config, self.visdir, self.dance_names, epoch_i, quants)
            model.train()
        self.schedular.step()



if __name__ == "__main__":
    main()
