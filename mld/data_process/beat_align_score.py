import numpy as np
import json
import os
from  scipy.ndimage import gaussian_filter as G
from scipy.signal import argrelextrema


music_root = '/mnt/disk_1/yufei/datasets_dev/AISTPP_processed/aistpp_test_full_wav'


def get_mb(key, length=None):
    path = os.path.join(music_root, key)
    with open(path) as f:
        #print(path)
        sample_dict = json.loads(f.read())
        if length is not None:
            beats = np.array(sample_dict['music_array'])[:, 53][:][:length]
        else:
            beats = np.array(sample_dict['music_array'])[:, 53]


        beats = beats.astype(bool)
        beat_axis = np.arange(len(beats))
        beat_axis = beat_axis[beats]
        return beat_axis


def calc_db(keypoints, name=''):
    keypoints = np.array(keypoints).reshape(-1, 24, 3)
    kinetic_vel = np.mean(np.sqrt(np.sum((keypoints[1:] - keypoints[:-1]) ** 2, axis=2)), axis=1)
    kinetic_vel = G(kinetic_vel, 5)
    motion_beats = argrelextrema(kinetic_vel, np.less)
    return motion_beats, len(kinetic_vel)


def BA(music_beats, motion_beats):
    ba = 0
    for bb in music_beats:
        ba +=  np.exp(-np.min((motion_beats[0] - bb)**2) / 2 / 9)
    return (ba / len(music_beats))

def calc_ba_score(root):

    # gt_list = []
    ba_scores = []

    for pkl in os.listdir(root):
        # print(pkl)
        if os.path.isdir(os.path.join(root, pkl)):
            continue
        joint3d = np.load(os.path.join(root, pkl), allow_pickle=True).item()['pred_position'][:, :]

        dance_beats, length = calc_db(joint3d, pkl)        
        music_beats = get_mb(pkl.split('.')[0] + '.json', length)

        ba_scores.append(BA(music_beats, dance_beats))
        
    return np.mean(ba_scores)

if __name__ == '__main__':
    pred_root = '/mnt/disk_1/yufei/experiments_music/aist++/step1/pkl/ep000000'
    print(calc_ba_score(pred_root))
