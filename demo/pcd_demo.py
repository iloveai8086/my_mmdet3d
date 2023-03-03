# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmdet3d.apis import inference_detector, init_model, show_result_meshlab

# python pcd_demo.py --pcd data/nuscenes/seq_0_frame_100.bin --config ../configs/centerpoint/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus.py --checkpoint ../checkpoints/centerpoint_01voxel_second_secfpn_circlenms_4x8_cyclic_20e_nus_20201001_135205-5db91e00.pth --show

# python demo/pcd_demo.py --pcd demo/data/nuscenes/seq_0_frame_100.bin --config configs/centerpoint/centerpoint_02pillar_second_secfpn_dcn_4x8_cyclic_20e_nus.py --checkpoint checkpoints/centerpoint_02pillar_second_secfpn_dcn_4x8_cyclic_20e_nus_20210815_202702-f03ab9e4.pth --show

def main():
    parser = ArgumentParser()
    parser.add_argument('--pcd', help='Point cloud file')
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    parser.add_argument(
        '--out-dir', type=str, default='demo', help='dir to save results')
    parser.add_argument(
        '--show',
        action='store_true',
        help='show online visualization results')
    parser.add_argument(
        '--snapshot',
        action='store_true',
        help='whether to save online visualization results')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    # test a single image
    result, data = inference_detector(model, args.pcd)
    # show the results
    show_result_meshlab(
        data,
        result,
        args.out_dir,
        args.score_thr,
        show=args.show,
        snapshot=args.snapshot,
        task='det')


if __name__ == '__main__':
    main()
