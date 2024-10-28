from music_data import quantized_metrics,calc_and_save_feats


if __name__ == '__main__':
    gt_root = 'data/aist_features_zero_start'
    pred_root = 'experiments/actor_critic/eval/pkl/ep000010'
    print('Calculating and saving features')
    calc_and_save_feats(gt_root)
    calc_and_save_feats(pred_root)

    print('Calculating metrics')
    print(gt_root)
    print(pred_root)
    print(quantized_metrics(pred_root, gt_root))