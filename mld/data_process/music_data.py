# the evaluate code borrows from Bailando
# https://github.com/lisiyao21/Bailando

import numpy as np
import os
import json
import torch.utils.data
from torch.utils.data import Dataset
from tqdm import tqdm
import shutil
import utils as feat_utils
from scipy import linalg
from multiprocessing import Pool

from PIL import Image
from features.kinetic import extract_kinetic_features
from features.manual_new import extract_manual_features
from functools import partial

# kinetic, manual
import os
import cv2
from keypoints2img import read_keypoints
pose_keypoints_num = 25
face_keypoints_num = 70
hand_left_keypoints_num = 21
hand_right_keypoints_num = 21
def normalize(feat, feat2):
    mean = feat.mean(axis=0)
    std = feat.std(axis=0)

    return (feat - mean) / (std + 1e-10), (feat2 - mean) / (std + 1e-10)

def visualize_json(fname_iter, image_dir, dance_name, dance_path, config, quant=None):
    j, fname = fname_iter
    json_file = os.path.join(dance_path, fname)
    img = Image.fromarray(read_keypoints(json_file, (config.width, config.height),
                                         remove_face_labels=False, basic_point_only=False))
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img = np.asarray(img)
    img_copy = np.array(img, copy=True)
    if quant is not None:
        cv2.putText(img_copy, str(quant[j]), (config.width-400, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2BGRA)
    # img[np.all(img == [0, 0, 0, 255], axis=2)] = [255, 255, 255, 0]
    img_copy = Image.fromarray(np.uint8(img_copy))
    img_copy.save(os.path.join(f'{image_dir}/{dance_name}', f'frame{j:06d}.png'))

def quantized_metrics(predicted_pkl_root, gt_pkl_root):

    pred_features_k = [np.load(os.path.join(predicted_pkl_root, 'kinetic_features', pkl)) for pkl in
                       os.listdir(os.path.join(predicted_pkl_root, 'kinetic_features'))]
    pred_features_m = [np.load(os.path.join(predicted_pkl_root, 'manual_features_new', pkl)) for pkl in
                       os.listdir(os.path.join(predicted_pkl_root, 'manual_features_new'))]

    gt_freatures_k = [np.load(os.path.join(gt_pkl_root, 'kinetic_features', pkl)) for pkl in
                      os.listdir(os.path.join(gt_pkl_root, 'kinetic_features'))]
    gt_freatures_m = [np.load(os.path.join(gt_pkl_root, 'manual_features_new', pkl)) for pkl in
                      os.listdir(os.path.join(gt_pkl_root, 'manual_features_new'))]

    pred_features_k = np.stack(pred_features_k)  # Nx72 p40
    pred_features_m = np.stack(pred_features_m)  # Nx32
    gt_freatures_k = np.stack(gt_freatures_k)  # N' x 72 N' >> N
    gt_freatures_m = np.stack(gt_freatures_m)  #

    gt_freatures_k, pred_features_k = normalize(gt_freatures_k, pred_features_k)
    gt_freatures_m, pred_features_m = normalize(gt_freatures_m, pred_features_m)

    fid_k = calc_fid(pred_features_k, gt_freatures_k)
    fid_m = calc_fid(pred_features_m, gt_freatures_m)

    div_k_gt = calculate_avg_distance(gt_freatures_k)
    div_m_gt = calculate_avg_distance(gt_freatures_m)
    div_k = calculate_avg_distance(pred_features_k)
    div_m = calculate_avg_distance(pred_features_m)

    if isinstance(fid_k,complex):
        fid_k = fid_k.real
    if isinstance(fid_m,complex):
        fid_m = fid_m.real

    metrics = {'fid_k': fid_k, 'fid_m': fid_m, 'div_k': div_k, 'div_m': div_m, 'div_k_gt': div_k_gt,
               'div_m_gt': div_m_gt}

    #metrics = {'fid_k': fid_k, 'fid_m': fid_m}
    return metrics


def calc_fid(kps_gen, kps_gt):
    # kps_gen = kps_gen[:20, :]

    mu_gen = np.mean(kps_gen, axis=0)
    sigma_gen = np.cov(kps_gen, rowvar=False)

    mu_gt = np.mean(kps_gt, axis=0)
    sigma_gt = np.cov(kps_gt, rowvar=False)

    mu1, mu2, sigma1, sigma2 = mu_gen, mu_gt, sigma_gen, sigma_gt

    diff = mu1 - mu2
    eps = 1e-5
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps

        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calc_diversity(feats):
    feat_array = np.array(feats)
    n, c = feat_array.shape
    diff = np.array([feat_array] * n) - feat_array.reshape(n, 1, c)
    return np.sqrt(np.sum(diff ** 2, axis=2)).sum() / n / (n - 1)


def calculate_avg_distance(feature_list, mean=None, std=None):
    feature_list = np.stack(feature_list)
    n = feature_list.shape[0]
    # normalize the scale
    if (mean is not None) and (std is not None):
        feature_list = (feature_list - mean) / std
    dist = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist += np.linalg.norm(feature_list[i] - feature_list[j])
    dist /= (n * n - n) / 2
    return dist


def calc_and_save_feats(root):
    if not os.path.exists(os.path.join(root, 'kinetic_features')):
        os.mkdir(os.path.join(root, 'kinetic_features'))
    if not os.path.exists(os.path.join(root, 'manual_features_new')):
        os.mkdir(os.path.join(root, 'manual_features_new'))

    # gt_list = []
    pred_list = []

    for pkl in os.listdir(root):

        if os.path.isdir(os.path.join(root, pkl)):
            continue
        joint3d = np.load(os.path.join(root, pkl), allow_pickle=True).item()['pred_position'][:1200, :]
        # print(extract_manual_features(joint3d.reshape(-1, 24, 3)))
        roott = joint3d[:1, :3]  # the root Tx72 (Tx(24x3))
        # print(roott)
        joint3d = joint3d - np.tile(roott, (1, 24))  # Calculate relative offset with respect to root
        # print('==============after fix root ============')
        # print(extract_manual_features(joint3d.reshape(-1, 24, 3)))
        # print('==============bla============')
        # print(extract_manual_features(joint3d.reshape(-1, 24, 3)))
        # np_dance[:, :3] = root
        np.save(os.path.join(root, 'kinetic_features', pkl), extract_kinetic_features(joint3d.reshape(-1, 24, 3)))
        np.save(os.path.join(root, 'manual_features_new', pkl), extract_manual_features(joint3d.reshape(-1, 24, 3)))



def calculate_metrics(pred_root):
    gt_root = '/mnt/disk_1/yufei/datasets_dev/aist_features_zero_start'
    calc_and_save_feats(gt_root)
    calc_and_save_feats(pred_root)
    metrics = quantized_metrics(pred_root,gt_root)
    return metrics


SMPL_JOINT_NAMES = [
    "root",
    "lhip", "rhip", "belly",
    "lknee", "rknee", "spine",
    "lankle", "rankle", "chest",
    "ltoes", "rtoes", "neck",
    "linshoulder", "rinshoulder",
    "head",  "lshoulder", "rshoulder",
    "lelbow", "relbow",
    "lwrist", "rwrist",
    "lhand", "rhand",
]
class MoSeq(Dataset):
    def __init__(self, dances):
        self.dances = dances

    def __len__(self):
        return len(self.dances)

    def __getitem__(self, index):
        # print(self.dances[index].shape)
        return self.dances[index]



def load_data_aist(data_dir, interval=120, move=40, rotmat=False, external_wav=None, external_wav_rate=1,
                   music_normalize=False, wav_padding=0):
    tot = 0
    music_data, dance_data = [], []
    input_names = []
    fnames = sorted(os.listdir(data_dir))



    if ".ipynb_checkpoints" in fnames:
        fnames.remove(".ipynb_checkpoints")
    for fname in fnames:

        path = os.path.join(data_dir, fname)
        with open(path) as f:
            # print(path)
            sample_dict = json.loads(f.read())
            np_music = np.array(sample_dict['music_array'])

            if external_wav is not None:
                wav_path = os.path.join(external_wav, fname.split('_')[-2] + '.json')
                # print('load from external wav!')
                with open(wav_path) as ff:
                    sample_dict_wav = json.loads(ff.read())
                    np_music = np.array(sample_dict_wav['music_array']).astype(np.float32)

            np_dance = np.array(sample_dict['dance_array'])

            if not rotmat:
                root = np_dance[:, :3]  # the root
                np_dance = np_dance - np.tile(root, (1, 24))  # Calculate relative offset with respect to root
                np_dance[:, :3] = root

            music_sample_rate = external_wav_rate if external_wav is not None else 1 #8

            # print('music_sample_rate', music_sample_rate)
            # print(music_sample_rate)
            if interval is not None:
                seq_len, dim = np_music.shape

                for i in range(0, seq_len, move):
                    i_sample = i // music_sample_rate
                    interval_sample = interval // music_sample_rate

                    music_sub_seq = np_music[i_sample: i_sample + interval_sample]
                    dance_sub_seq = np_dance[i: i + interval]

                    if len(music_sub_seq) == interval_sample and len(dance_sub_seq) == interval:
                        padding_sample = wav_padding // music_sample_rate
                        # Add paddings/context of music
                        music_sub_seq_pad = np.zeros((interval_sample + padding_sample * 2, dim),
                                                     dtype=music_sub_seq.dtype)

                        if padding_sample > 0:
                            music_sub_seq_pad[padding_sample:-padding_sample] = music_sub_seq
                            start_sample = padding_sample if i_sample > padding_sample else i_sample
                            end_sample = padding_sample if i_sample + interval_sample + padding_sample < seq_len else seq_len - (
                                        i_sample + interval_sample)
                            # print(end_sample)
                            music_sub_seq_pad[padding_sample - start_sample:padding_sample] = np_music[
                                                                                              i_sample - start_sample:i_sample]
                            if end_sample == padding_sample:
                                music_sub_seq_pad[-padding_sample:] = np_music[
                                                                      i_sample + interval_sample:i_sample + interval_sample + end_sample]
                            else:
                                music_sub_seq_pad[-padding_sample:-padding_sample + end_sample] = np_music[
                                                                                                  i_sample + interval_sample:i_sample + interval_sample + end_sample]
                        else:

                            music_sub_seq_pad = music_sub_seq
                        music_data.append(music_sub_seq_pad)
                        dance_data.append(dance_sub_seq)
                        input_names.append(fname)
                        tot += 1
                        # if tot > 1:
                        #     break
            else:
                music_data.append(np_music)
                dance_data.append(np_dance)

            # if tot > 1:
            #     break

            # tot += 1
            # if tot > 100:
            #     break

    # print(tot) 34271

    music_np = np.stack(music_data).reshape(-1, music_data[0].shape[1])
    music_mean = music_np.mean(0)
    music_std = music_np.std(0)
    music_std[(np.abs(music_mean) < 1e-5) & (np.abs(music_std) < 1e-5)] = 1

    # music_data_norm = [ (music_sub_seq - music_mean) / (music_std + 1e-10) for music_sub_seq in music_data ]
    # print(music_np)

    if music_normalize:
        print('calculating norm mean and std')
        music_data_norm = [(music_sub_seq - music_mean) / (music_std + 1e-10) for music_sub_seq in music_data]
        with open('/mnt/lustressd/lisiyao1/dance_experiements/music_norm.json', 'w') as fff:
            sample_dict = {
                'music_mean': music_mean.tolist(),  # musics[idx+i],
                'music_std': music_std.tolist()
            }
            # print(sample_dict)
            json.dump(sample_dict, fff)
    else:
        music_data_norm = music_data

    return music_data_norm, dance_data, input_names
    # , [fn.replace('.json', '') for fn in fnames]

def prepare_dataloader(music_data, dance_data, batch_size):
    data_loader = torch.utils.data.DataLoader(
        MoDaSeq(music_data, dance_data),
        num_workers=8,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
                # collate_fn=paired_collate_fn,
    )

    return data_loader


def load_test_data_aist(data_dir, rotmat, move, external_wav=None, external_wav_rate=1, music_normalize=False,
                        wav_padding=0):
    tot = 0
    input_names = []

    music_data, dance_data = [], []
    fnames = sorted(os.listdir(data_dir))
    # print(fnames)
    #fnames = fnames[:10]  # For debug
    for fname in fnames:
        path = os.path.join(data_dir, fname)
        with open(path) as f:
            # print(path)
            sample_dict = json.loads(f.read())
            np_music = np.array(sample_dict['music_array'])
            if external_wav is not None:
                # print('load from external wav!')
                wav_path = os.path.join(external_wav, fname.split('_')[-2] + '.json')
                with open(wav_path) as ff:
                    sample_dict_wav = json.loads(ff.read())
                    np_music = np.array(sample_dict_wav['music_array'])

            if 'dance_array' in sample_dict:
                np_dance = np.array(sample_dict['dance_array'])
                if not rotmat:
                    root = np_dance[:, :3]  # the root
                    np_dance = np_dance - np.tile(root, (1, 24))  # Calculate relative offset with respect to root
                    np_dance[:, :3] = root

                for kk in range((len(np_dance) // move + 1) * move - len(np_dance)):
                    np_dance = np.append(np_dance, np_dance[-1:], axis=0)

                dance_data.append(np_dance)
            # print('Music data shape: ', np_music.shape)
            else:
                np_dance = None
                dance_data = None
            music_move = external_wav_rate if external_wav is not None else move

            # zero padding left
            for kk in range(wav_padding):
                np_music = np.append(np.zeros_like(np_music[-1:]), np_music, axis=0)
            # fully devisable
            for kk in range((len(np_music) // music_move + 1) * music_move - len(np_music)):
                np_music = np.append(np_music, np_music[-1:], axis=0)
            # zero padding right
            for kk in range(wav_padding):
                np_music = np.append(np_music, np.zeros_like(np_music[-1:]), axis=0)

            music_data.append(np_music)
            input_names.append(fname)
            # tot += 1
            # if tot == 3:
            #     break
    # if music_normalize:
    if False:
        with open('/mnt/lustressd/lisiyao1/dance_experiements/music_norm.json') as fff:
            sample_dict = json.loads(fff.read())
            music_mean = np.array(sample_dict['music_mean'])
            music_std = np.array(sample_dict['music_std'])
        music_std[(np.abs(music_mean) < 1e-5) & (np.abs(music_std) < 1e-5)] = 1

        music_data_norm = [(music_sub_seq - music_mean) / (music_std + 1e-10) for music_sub_seq in music_data]
    else:
        music_data_norm = music_data

    return music_data_norm, dance_data, input_names

class MoDaSeq(Dataset):
    def __init__(self, musics, dances=None):
        if dances is not None:
            assert (len(musics) == len(dances)), \
                'the number of dances should be equal to the number of musics'
        self.musics = musics
        self.dances = dances
        # if clip_names is not None:
        # self.clip_names = clip_names

    def __len__(self):
        return len(self.musics)

    def __getitem__(self, index):
        if self.dances is not None:
            # if self.clip_names is not None:
            #     return self.musics[index], self.dances[index], self.clip_names[index]
            # else:
            return self.musics[index], self.dances[index],
        else:
            return self.musics[index]


def write2pkl(dances, dance_names, config, expdir, epoch, rotmat):
    epoch = int(epoch)
    # print(len(dances))
    # print(len(dance_names))
    # exit(0)
    assert len(dances) == len(dance_names), \
        "number of generated dance != number of dance_names"

    if not os.path.exists(os.path.join(expdir, "pkl")):
        os.makedirs(os.path.join(expdir, "pkl"))

    ep_path = os.path.join(expdir, "pkl", f"ep{epoch:06d}")

    if not os.path.exists(ep_path):
        os.makedirs(ep_path)

    # print("Writing Json...")
    for i in tqdm(range(len(dances)), desc='Generating Jsons'):
        # if rotmat:
        #     mat, trans = dances[i]
        #     pkl_data = {"pred_motion": mat, "pred_trans": trans}
        # else:
        np_dance = dances[i]
        pkl_data = {"pred_position": np_dance}

        dance_path = os.path.join(ep_path, dance_names[i] + '.pkl')
        # if not os.path.exists(dance_path):
        #     os.makedirs(dance_path)

        # with open(dance_path, 'w') as f:
        np.save(dance_path, pkl_data)
    return ep_path


def pose_code2pkl(pcodes, dance_names, config, expdir, epoch):
    epoch = int(epoch)
    # print(len(pcodes))
    # print(len(dance_names))
    assert len(pcodes) == len(dance_names), \
        "number of generated dance != number of dance_names"

    if not os.path.exists(os.path.join(expdir, "pose_codes")):
        os.makedirs(os.path.join(expdir, "pose_codes"))

    ep_path = os.path.join(expdir, "pose_codes", f"ep{epoch:06d}")

    if not os.path.exists(ep_path):
        os.makedirs(ep_path)

    # print("Writing Json...")
    for i in tqdm(range(len(pcodes)), desc='writing pose code'):
        # if rotmat:
        #     mat, trans = dances[i]
        #     pkl_data = {"pred_motion": mat, "pred_trans": trans}
        # else:

        name = dance_names[i]
        pcode = pcodes[name]

        pkl_data = {"pcodes_up": pcode[0], "pcodes_down": pcode[1]}

        dance_path = os.path.join(ep_path, name + '.pkl')
        # if not os.path.exists(dance_path):
        #     os.makedirs(dance_path)

        # with open(dance_path, 'w') as f:
        np.save(dance_path, pkl_data)


def write2json(dances, dance_names, config, expdir, epoch):
    epoch = int(epoch)
    assert len(dances) == len(dance_names), \
        "number of generated dance != number of dance_names"

    ep_path = os.path.join(expdir, "jsons", f"ep{epoch:06d}")

    if not os.path.exists(ep_path):
        os.makedirs(ep_path)

    # print("Writing Json...")
    for i in tqdm(range(len(dances)), desc='Generating Jsons'):
        num_poses = dances[i].shape[0]
        dances[i] = dances[i].reshape(num_poses, pose_keypoints_num, 2)
        dance_path = os.path.join(ep_path, dance_names[i])
        if not os.path.exists(dance_path):
            os.makedirs(dance_path)

        for j in range(num_poses):
            frame_dict = {'version': 1.2}
            # 2-D key points
            pose_keypoints_2d = []
            # Random values for the below key points
            face_keypoints_2d = []
            hand_left_keypoints_2d = []
            hand_right_keypoints_2d = []
            # 3-D key points
            pose_keypoints_3d = []
            face_keypoints_3d = []
            hand_left_keypoints_3d = []
            hand_right_keypoints_3d = []

            keypoints = dances[i][j]
            for k, keypoint in enumerate(keypoints):
                x = (keypoint[0] + 1) * 0.5 * config.width
                y = (keypoint[1] + 1) * 0.5 * config.height
                score = 0.8
                if k < pose_keypoints_num:
                    pose_keypoints_2d.extend([x, y, score])
                elif k < pose_keypoints_num + face_keypoints_num:
                    face_keypoints_2d.extend([x, y, score])
                elif k < pose_keypoints_num + face_keypoints_num + hand_left_keypoints_num:
                    hand_left_keypoints_2d.extend([x, y, score])
                else:
                    hand_right_keypoints_2d.extend([x, y, score])

            people_dicts = []
            people_dict = {'pose_keypoints_2d': pose_keypoints_2d,
                           'face_keypoints_2d': face_keypoints_2d,
                           'hand_left_keypoints_2d': hand_left_keypoints_2d,
                           'hand_right_keypoints_2d': hand_right_keypoints_2d,
                           'pose_keypoints_3d': pose_keypoints_3d,
                           'face_keypoints_3d': face_keypoints_3d,
                           'hand_left_keypoints_3d': hand_left_keypoints_3d,
                           'hand_right_keypoints_3d': hand_right_keypoints_3d}
            people_dicts.append(people_dict)
            frame_dict['people'] = people_dicts
            frame_json = json.dumps(frame_dict)
            with open(os.path.join(dance_path, f'ep{epoch:06d}_frame{j:06d}_kps.json'), 'w') as f:
                f.write(frame_json)


def visualize(config, the_dance_names, expdir, epoch, quants=None, worker_num=16):
    epoch = int(epoch)
    json_dir = os.path.join(expdir, "jsons", f"ep{epoch:06d}")

    image_dir = os.path.join(expdir, "imgs", f"ep{epoch:06d}")

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    dance_names = sorted(os.listdir(json_dir))
    dance_names = the_dance_names

    # print(quants)
    quant_list = None

    # print("Visualizing")
    for i, dance_name in enumerate(tqdm(dance_names, desc='Generating Images')):
        dance_path = os.path.join(json_dir, dance_name)
        fnames = sorted(os.listdir(dance_path))
        if not os.path.exists(f'{image_dir}/{dance_name}'):
            os.makedirs(f'{image_dir}/{dance_name}')
        if quants is not None:
            if isinstance(quants[dance_name], tuple):
                quant_lists = []
                for qs in quants[dance_name]:
                    downsample_rate = max(len(fnames) // len(qs), 1)
                    quant_lists.append(qs.repeat(downsample_rate).tolist())
                quant_list = [tuple(qlist[ii] for qlist in quant_lists) for ii in range(len(quant_lists[0]))]
            # while len(quant_list) < len(dance_names):
            # print(quants)
            # print(len(fnames), len(quants[dance_name]))
            else:
                downsample_rate = max(len(fnames) // len(quants[dance_name]), 1)
                quant_list = quants[dance_name].repeat(downsample_rate).tolist()
            while len(quant_list) < len(dance_names):
                quant_list.append(quant_list[-1])

                # Visualize json in parallel
        pool = Pool(worker_num)
        partial_func = partial(visualize_json, image_dir=image_dir,
                               dance_name=dance_name, dance_path=dance_path, config=config, quant=quant_list)
        pool.map(partial_func, enumerate(fnames))
        pool.close()
        pool.join()


def img2video(expdir, epoch, audio_path=None):
    video_dir = os.path.join(expdir, "videos", f"ep{epoch:06d}")
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    image_dir = os.path.join(expdir, "imgs", f"ep{epoch:06d}")

    dance_names = sorted(os.listdir(image_dir))
    audio_dir = "/mnt/disk_1/yufei/datasets_dev/aist_plusplus_final/all_musics"

    music_names = sorted(os.listdir(audio_dir))

    for dance in tqdm(dance_names, desc='Generating Videos'):
        # pdb.set_trace()
        name = dance.split(".")[0]
        cmd = f"ffmpeg -r 60 -i {image_dir}/{dance}/frame%06d.png -vb 20M -vcodec mpeg4 -y {video_dir}/{name}.mp4 -loglevel quiet"
        # cmd = f"ffmpeg -r 60 -i {image_dir}/{dance}/%05d.png -vb 20M -vcodec qtrle -y {video_dir}/{name}.mov -loglevel quiet"

        os.system(cmd)

        name1 = name.replace('cAll', 'c02')

        if 'cAll' in name:
            music_name = name[-9:-5] + '.wav'
        else:
            music_name = name + '.mp3'
            audio_dir = 'extra/'
            music_names = sorted(os.listdir(audio_dir))

        if music_name in music_names:
            print('combining audio!')
            audio_dir_ = os.path.join(audio_dir, music_name)
            print(audio_dir_)
            name_w_audio = name + "_audio"
            cmd_audio = f"ffmpeg -i {video_dir}/{name}.mp4 -i {audio_dir_} -map 0:v -map 1:a -c:v copy -shortest -y {video_dir}/{name_w_audio}.mp4 -loglevel quiet"
            os.system(cmd_audio)


def visualizeAndWrite(results, config, expdir, dance_names, epoch, quants=None):
    if config.rotmat:
        smpl = SMPL(model_path=config.smpl_dir, gender='MALE', batch_size=1)
    np_dances = []
    np_dances_original = []
    dance_datas = []
    if config.data.name == "aist":

        for i in range(len(results)):
            np_dance = results[i][0].data.cpu().numpy()

            if config.rotmat:
                print('Use SMPL!')
                root = np_dance[:, :3]
                rotmat = np_dance[:, 3:].reshape([-1, 3, 3])

                # write2pkl((rotmat, root), dance_names[i], config.testing, expdir, epoch, rotmat=True)

                rotmat = get_closest_rotmat(rotmat)
                smpl_poses = rotmat2aa(rotmat).reshape(-1, 24, 3)
                np_dance = smpl.forward(
                    global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
                    body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
                    transl=torch.from_numpy(root).float(),
                ).joints.detach().numpy()[:, 0:24, :]
                b = np_dance.shape[0]
                np_dance = np_dance.reshape(b, -1)
                dance_datas.append(np_dance)
            # print(np_dance.shape)
            else:
                # if args.use_mean_pose:
                #     print('We use mean pose!')
                #     np_dance += mean_pose
                root = np_dance[:, :3]
                np_dance = np_dance + np.tile(root, (1, 24))
                np_dance[:, :3] = root

                dance_datas.append(np_dance)
                # write2pkl(np_dance, dance_names[i], config.testing, expdir, epoch, rotmat=True)

            root = np_dance[:, :3]
            # np_dance = np_dance + np.tile(root, (1, 24))
            np_dance[:, :3] = root
            # np_dance[2:-2] = (np_dance[:-4] + np_dance[1:-3] + np_dance[2:-2] +  np_dance[3:-1] + np_dance[4:]) / 5.0
            np_dances_original.append(np_dance)

            b, c = np_dance.shape
            np_dance = np_dance.reshape([b, c // 3, 3])
            # np_dance2 = np_dance[:, :, :2] / 2 - 0.5
            # np_dance2[:, :, 1] = np_dance2[:, :, 1]
            np_dance2 = np_dance[:, :, :2] / 1.5
            np_dance2[:, :, 0] /= 2.2
            np_dance_trans = np.zeros([b, 25, 2]).copy()

            # head
            np_dance_trans[:, 0] = np_dance2[:, 12]

            # neck
            np_dance_trans[:, 1] = np_dance2[:, 9]

            # left up
            np_dance_trans[:, 2] = np_dance2[:, 16]
            np_dance_trans[:, 3] = np_dance2[:, 18]
            np_dance_trans[:, 4] = np_dance2[:, 20]

            # right up
            np_dance_trans[:, 5] = np_dance2[:, 17]
            np_dance_trans[:, 6] = np_dance2[:, 19]
            np_dance_trans[:, 7] = np_dance2[:, 21]

            np_dance_trans[:, 8] = np_dance2[:, 0]

            np_dance_trans[:, 9] = np_dance2[:, 1]
            np_dance_trans[:, 10] = np_dance2[:, 4]
            np_dance_trans[:, 11] = np_dance2[:, 7]

            np_dance_trans[:, 12] = np_dance2[:, 2]
            np_dance_trans[:, 13] = np_dance2[:, 5]
            np_dance_trans[:, 14] = np_dance2[:, 8]

            np_dance_trans[:, 15] = np_dance2[:, 15]
            np_dance_trans[:, 16] = np_dance2[:, 15]
            np_dance_trans[:, 17] = np_dance2[:, 15]
            np_dance_trans[:, 18] = np_dance2[:, 15]

            np_dance_trans[:, 19] = np_dance2[:, 11]
            np_dance_trans[:, 20] = np_dance2[:, 11]
            np_dance_trans[:, 21] = np_dance2[:, 8]

            np_dance_trans[:, 22] = np_dance2[:, 10]
            np_dance_trans[:, 23] = np_dance2[:, 10]
            np_dance_trans[:, 24] = np_dance2[:, 7]

            np_dances.append(np_dance_trans.reshape([b, 25 * 2]))
    else:
        for i in range(len(results)):
            np_dance = results[i][0].data.cpu().numpy()
            root = np_dance[:, 2 * 8:2 * 9]
            np_dance = np_dance + np.tile(root, (1, 25))
            np_dance[:, 2 * 8:2 * 9] = root
            np_dances.append(np_dance)
    ep_path = write2pkl(dance_datas, dance_names, config.testing, expdir, epoch, rotmat=config.rotmat)
    pose_code2pkl(quants, dance_names, config.testing, expdir, epoch)
    write2json(np_dances, dance_names, config.testing, expdir, epoch)
    visualize(config.testing, dance_names, expdir, epoch, quants)
    img2video(expdir,epoch)

    json_dir = os.path.join(expdir, "jsons", f"ep{epoch:06d}")
    img_dir = os.path.join(expdir, "imgs", f"ep{epoch:06d}")
    if os.path.exists(json_dir):
        shutil.rmtree(json_dir)
    # if os.path.exists(img_dir):
    #     shutil.rmtree(img_dir)
    return ep_path

def visualizeAndWrite_novis(results, config, expdir, dance_names, epoch, quants=None):
    if config.rotmat:
        smpl = SMPL(model_path=config.smpl_dir, gender='MALE', batch_size=1)
    np_dances = []
    np_dances_original = []
    dance_datas = []
    if config.data.name == "aist":

        for i in range(len(results)):
            np_dance = results[i][0].data.cpu().numpy()

            if config.rotmat:
                print('Use SMPL!')
                root = np_dance[:, :3]
                rotmat = np_dance[:, 3:].reshape([-1, 3, 3])

                # write2pkl((rotmat, root), dance_names[i], config.testing, expdir, epoch, rotmat=True)

                rotmat = get_closest_rotmat(rotmat)
                smpl_poses = rotmat2aa(rotmat).reshape(-1, 24, 3)
                np_dance = smpl.forward(
                    global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
                    body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
                    transl=torch.from_numpy(root).float(),
                ).joints.detach().numpy()[:, 0:24, :]
                b = np_dance.shape[0]
                np_dance = np_dance.reshape(b, -1)
                dance_datas.append(np_dance)
            # print(np_dance.shape)
            else:
                # if args.use_mean_pose:
                #     print('We use mean pose!')
                #     np_dance += mean_pose
                root = np_dance[:, :3]
                np_dance = np_dance + np.tile(root, (1, 24))
                np_dance[:, :3] = root

                dance_datas.append(np_dance)
                # write2pkl(np_dance, dance_names[i], config.testing, expdir, epoch, rotmat=True)

            root = np_dance[:, :3]
            # np_dance = np_dance + np.tile(root, (1, 24))
            np_dance[:, :3] = root
            # np_dance[2:-2] = (np_dance[:-4] + np_dance[1:-3] + np_dance[2:-2] +  np_dance[3:-1] + np_dance[4:]) / 5.0
            np_dances_original.append(np_dance)

            b, c = np_dance.shape
            np_dance = np_dance.reshape([b, c // 3, 3])
            # np_dance2 = np_dance[:, :, :2] / 2 - 0.5
            # np_dance2[:, :, 1] = np_dance2[:, :, 1]
            np_dance2 = np_dance[:, :, :2] / 1.5
            np_dance2[:, :, 0] /= 2.2
            np_dance_trans = np.zeros([b, 25, 2]).copy()

            # head
            np_dance_trans[:, 0] = np_dance2[:, 12]

            # neck
            np_dance_trans[:, 1] = np_dance2[:, 9]

            # left up
            np_dance_trans[:, 2] = np_dance2[:, 16]
            np_dance_trans[:, 3] = np_dance2[:, 18]
            np_dance_trans[:, 4] = np_dance2[:, 20]

            # right up
            np_dance_trans[:, 5] = np_dance2[:, 17]
            np_dance_trans[:, 6] = np_dance2[:, 19]
            np_dance_trans[:, 7] = np_dance2[:, 21]

            np_dance_trans[:, 8] = np_dance2[:, 0]

            np_dance_trans[:, 9] = np_dance2[:, 1]
            np_dance_trans[:, 10] = np_dance2[:, 4]
            np_dance_trans[:, 11] = np_dance2[:, 7]

            np_dance_trans[:, 12] = np_dance2[:, 2]
            np_dance_trans[:, 13] = np_dance2[:, 5]
            np_dance_trans[:, 14] = np_dance2[:, 8]

            np_dance_trans[:, 15] = np_dance2[:, 15]
            np_dance_trans[:, 16] = np_dance2[:, 15]
            np_dance_trans[:, 17] = np_dance2[:, 15]
            np_dance_trans[:, 18] = np_dance2[:, 15]

            np_dance_trans[:, 19] = np_dance2[:, 11]
            np_dance_trans[:, 20] = np_dance2[:, 11]
            np_dance_trans[:, 21] = np_dance2[:, 8]

            np_dance_trans[:, 22] = np_dance2[:, 10]
            np_dance_trans[:, 23] = np_dance2[:, 10]
            np_dance_trans[:, 24] = np_dance2[:, 7]

            np_dances.append(np_dance_trans.reshape([b, 25 * 2]))
    else:
        for i in range(len(results)):
            np_dance = results[i][0].data.cpu().numpy()
            root = np_dance[:, 2 * 8:2 * 9]
            np_dance = np_dance + np.tile(root, (1, 25))
            np_dance[:, 2 * 8:2 * 9] = root
            np_dances.append(np_dance)
    ep_path = write2pkl(dance_datas, dance_names, config.testing, expdir, epoch, rotmat=config.rotmat)

    json_dir = os.path.join(expdir, "jsons", f"ep{epoch:06d}")
    img_dir = os.path.join(expdir, "imgs", f"ep{epoch:06d}")
    if os.path.exists(json_dir):
        shutil.rmtree(json_dir)
    # if os.path.exists(img_dir):
    #     shutil.rmtree(img_dir)
    return ep_path