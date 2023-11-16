import argparse
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # specify which GPU(s) to be used
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from external_src.group_free_3D.models import parse_predictions
from external_src.group_free_3D.models.ap_helper import dump_predictions, dump_ground_truth
from external_src.group_free_3D.models import GroupFreeDetector, get_loss
from external_src.group_free_3D.utils import setup_logger

DATASET_SPLIT = 'val'
OUTPUT_GT = False


def parse_option():
    parser = argparse.ArgumentParser()
    # Eval
    parser.add_argument('--checkpoint_path', default='external_src/group_free_3D/pretrained_model/ckpt_epoch_last.pth',
                        help='Model checkpoint path [default: None]')
    parser.add_argument('--avg_times', default=5, type=int, help='Average times')
    parser.add_argument("--rng_seed", type=int, default=0, help='manual seed')
    parser.add_argument('--dump_dir', default='dump', help='Dump dir to save sample outputs [default: None]')
    parser.add_argument('--use_old_type_nms', action='store_true', help='Use old type of NMS, IoBox2Area.')
    parser.add_argument('--nms_iou', type=float, default=0.25, help='NMS IoU threshold. [default: 0.25]')
    parser.add_argument('--conf_thresh', type=float, default=0.0,
                        help='Filter out predictions with obj prob less than it. [default: 0.05]')
    parser.add_argument('--ap_iou_thresholds', type=float, default=[0.25, 0.5], nargs='+',
                        help='A list of AP IoU thresholds [default: 0.25,0.5]')
    parser.add_argument('--faster_eval', action='store_true',
                        help='Faster evaluation by skippling empty bounding box removal.')
    parser.add_argument('--shuffle_dataset', action='store_true', help='Shuffle the dataset (random order).')

    # Model
    parser.add_argument('--width', default=1, type=int, help='backbone width')
    parser.add_argument('--num_target', type=int, default=256, help='Proposal number [default: 256]')
    parser.add_argument('--sampling', default='kps', type=str, help='Query points sampling method (kps, fps)')

    # Transformer
    parser.add_argument('--nhead', default=8, type=int, help='multi-head number')
    parser.add_argument('--num_decoder_layers', default=6, type=int, help='number of decoder layers')
    parser.add_argument('--dim_feedforward', default=2048, type=int, help='dim_feedforward')
    parser.add_argument('--transformer_dropout', default=0.1, type=float, help='transformer_dropout')
    parser.add_argument('--transformer_activation', default='relu', type=str, help='transformer_activation')
    parser.add_argument('--self_position_embedding', default='loc_learned', type=str,
                        help='position_embedding in self attention (none, xyz_learned, loc_learned)')
    parser.add_argument('--cross_position_embedding', default='xyz_learned', type=str,
                        help='position embedding in cross attention (none, xyz_learned)')

    # Loss
    parser.add_argument('--query_points_generator_loss_coef', default=0.8, type=float)
    parser.add_argument('--obj_loss_coef', default=0.1, type=float, help='Loss weight for objectness loss')
    parser.add_argument('--box_loss_coef', default=1, type=float, help='Loss weight for box loss')
    parser.add_argument('--sem_cls_loss_coef', default=0.1, type=float, help='Loss weight for classification loss')
    parser.add_argument('--center_loss_type', default='smoothl1', type=str, help='(smoothl1, l1)')
    parser.add_argument('--center_delta', default=0.4, type=float, help='delta for smoothl1 loss in center loss')  # TODO maybe revert to 1.0
    parser.add_argument('--size_loss_type', default='smoothl1', type=str, help='(smoothl1, l1)')
    parser.add_argument('--size_delta', default=1.0, type=float, help='delta for smoothl1 loss in size loss')
    parser.add_argument('--heading_loss_type', default='smoothl1', type=str, help='(smoothl1, l1)')
    parser.add_argument('--heading_delta', default=1.0, type=float, help='delta for smoothl1 loss in heading loss')
    parser.add_argument('--query_points_obj_topk', default=4, type=int, help='query_points_obj_topk')
    parser.add_argument('--size_cls_agnostic', action='store_true', help='Use class-agnostic size prediction.')

    # Data
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 8]')  # one gpu 8 two gpus 16
    parser.add_argument('--dataset', default='OR_4D', help='Dataset name. sunrgbd or scannet. [default: scannet]')
    parser.add_argument('--num_point', type=int, default=200000, help='Point Number [default: 50000]')  # maybe use 50000
    parser.add_argument('--data_root', default='data', help='data root path')
    parser.add_argument('--use_height', action='store_true', help='Use height signal in input.')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')

    args, unparsed = parser.parse_known_args()

    return args


def get_loader(args):
    # Create Dataset and Dataloader
    if args.dataset == 'OR_4D':
        from external_src.group_free_3D.OR_4D.OR_4D_detection_dataset import OR_4DDetectionDataset
        from external_src.group_free_3D.OR_4D.model_util_OR_4D import OR_4DDatasetConfig

        DATASET_CONFIG = OR_4DDatasetConfig()
        TEST_DATASET = OR_4DDetectionDataset(DATASET_SPLIT, num_points=args.num_point,
                                             augment=False,
                                             use_color=True,
                                             use_height=True if args.use_height else False,
                                             data_root=args.data_root)
    else:
        raise NotImplementedError(f'Unknown dataset {args.dataset}. Exiting...')

    logger.info(str(len(TEST_DATASET)))

    TEST_DATALOADER = DataLoader(TEST_DATASET,
                                 batch_size=args.batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers,
                                 pin_memory=True)
    return TEST_DATALOADER, DATASET_CONFIG


def get_model(args, DATASET_CONFIG):
    if args.use_height:
        num_input_channel = 4
    else:
        num_input_channel = 3

    model = GroupFreeDetector(num_class=DATASET_CONFIG.num_class,
                              num_heading_bin=DATASET_CONFIG.num_heading_bin,
                              num_size_cluster=DATASET_CONFIG.num_size_cluster,
                              mean_size_arr=DATASET_CONFIG.mean_size_arr,
                              input_feature_dim=num_input_channel,
                              width=args.width,
                              num_proposal=args.num_target,
                              sampling=args.sampling,
                              dropout=args.transformer_dropout,
                              activation=args.transformer_activation,
                              nhead=args.nhead,
                              num_decoder_layers=args.num_decoder_layers,
                              dim_feedforward=args.dim_feedforward,
                              self_position_embedding=args.self_position_embedding,
                              cross_position_embedding=args.cross_position_embedding,
                              size_cls_agnostic=True if args.size_cls_agnostic else False)

    criterion = get_loss
    return model, criterion


def load_checkpoint(args, model):
    # Load checkpoint if there is any
    if args.checkpoint_path is not None and os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        state_dict = checkpoint['model']
        save_path = checkpoint.get('save_path', 'none')
        for k in list(state_dict.keys()):
            state_dict[k[len("module."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        model.load_state_dict(state_dict)
        logger.info(f"{args.checkpoint_path} loaded successfully!!!")

        del checkpoint
        torch.cuda.empty_cache()
    else:
        raise FileNotFoundError
    return save_path


def infer(test_loader, DATASET_CONFIG, CONFIG_DICT, AP_IOU_THRESHOLDS, model, criterion, args, output_gt=False):
    prefixes = ['last_']

    model.eval()  # set model to eval mode (for bn and dp)

    for batch_idx, batch_data_label in enumerate(test_loader):
        scan_names = batch_data_label.pop('scan_name')
        if output_gt:
            dump_ground_truth(batch_data_label, DATASET_CONFIG, scan_names)
            continue
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].cuda(non_blocking=True)

        # Forward pass
        inputs = {'point_clouds': batch_data_label['point_clouds']}
        with torch.no_grad():
            end_points = model(inputs)

        # Compute loss
        for key in batch_data_label:
            assert (key not in end_points)
            end_points[key] = batch_data_label[key]

        loss, end_points = criterion(end_points, DATASET_CONFIG,
                                     num_decoder_layers=args.num_decoder_layers,
                                     query_points_generator_loss_coef=args.query_points_generator_loss_coef,
                                     obj_loss_coef=args.obj_loss_coef,
                                     box_loss_coef=args.box_loss_coef,
                                     sem_cls_loss_coef=args.sem_cls_loss_coef,
                                     query_points_obj_topk=args.query_points_obj_topk,
                                     center_loss_type=args.center_loss_type,
                                     center_delta=args.center_delta,
                                     size_loss_type=args.size_loss_type,
                                     size_delta=args.size_delta,
                                     heading_loss_type=args.heading_loss_type,
                                     heading_delta=args.heading_delta,
                                     size_cls_agnostic=args.size_cls_agnostic)
        for prefix in prefixes:
            batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT, prefix,
                                                   size_cls_agnostic=args.size_cls_agnostic)
            dump_predictions(end_points, DATASET_CONFIG, prefix, scan_names)


def eval(args):
    test_loader, DATASET_CONFIG = get_loader(args)
    n_data = len(test_loader.dataset)
    logger.info(f"length of testing dataset: {n_data}")

    model, criterion = get_model(args, DATASET_CONFIG)
    logger.info(str(model))
    load_checkpoint(args, model)
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        logger.info("Let's use %d GPUs!" % (torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    # Used for AP calculation
    CONFIG_DICT = {'remove_empty_box': (not args.faster_eval), 'use_3d_nms': True, 'nms_iou': args.nms_iou,
                   'use_old_type_nms': args.use_old_type_nms, 'cls_nms': True,
                   'per_class_proposal': True,
                   'conf_thresh': args.conf_thresh, 'dataset_config': DATASET_CONFIG}

    infer(test_loader, DATASET_CONFIG, CONFIG_DICT, args.ap_iou_thresholds,
          model, criterion, args, output_gt=OUTPUT_GT)


if __name__ == '__main__':
    print(f'USING DATASET_SPLIT {DATASET_SPLIT} - GT:{OUTPUT_GT}')
    opt = parse_option()

    opt.dump_dir = os.path.join(opt.dump_dir, f'eval_{opt.dataset}_{int(time.time())}_{np.random.randint(100000000)}')
    logger = setup_logger(output=opt.dump_dir, name="eval")

    eval(opt)
