from __future__ import print_function, division
import sys
sys.path.append('core')
import os
import cv2

import argparse
import time
import logging
import numpy as np
import torch
from tqdm import tqdm
#from raft_stereo import RAFTStereo, autocast
import core.AsymKD_datasets as datasets
from core.utils import InputPadder
from segment_anything import  sam_model_registry, SamPredictor
from AsymKD.dpt import AsymKD_DepthAnything
import torch.nn as nn
from depth_anything_for_evaluate.dpt import DepthAnything

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_errors(flow_gt, flow_preds, valid_arr):
    """Compute metrics for 'pred' compared to 'gt'

    Args:
        gt (numpy.ndarray): Ground truth values
        pred (numpy.ndarray): Predicted values

        gt.shape should be equal to pred.shape

    Returns:
        dict: Dictionary containing the following metrics:
            'a1': Delta1 accuracy: Fraction of pixels that are within a scale factor of 1.25
            'a2': Delta2 accuracy: Fraction of pixels that are within a scale factor of 1.25^2
            'a3': Delta3 accuracy: Fraction of pixels that are within a scale factor of 1.25^3
            'abs_rel': Absolute relative error
            'rmse': Root mean squared error
            'log_10': Absolute log10 error
            'sq_rel': Squared relative error
            'rmse_log': Root mean squared error on the log scale
            'silog': Scale invariant log error
    """
    a1_arr = []
    a2_arr = []
    a3_arr = []
    abs_rel_arr = []
    rmse_arr = []
    log_10_arr = []
    rmse_log_arr = []
    silog_arr = []
    sq_rel_arr = []
    min_depth_eval = 0.001
    max_depth_eval = 80

    

    for gt, pred, valid in zip(flow_gt,flow_preds,valid_arr):
        
        gt = gt.squeeze().cpu().numpy()
        pred = pred.squeeze().cpu().numpy()
        valid = valid.squeeze().cpu()
        pred[pred < min_depth_eval] = min_depth_eval
        pred[pred > max_depth_eval] = max_depth_eval
        pred[np.isinf(pred)] = max_depth_eval
        pred[np.isnan(pred)] = min_depth_eval
        # print(gt.max(),gt.min())
        # print(pred.max(),pred.min())
        gt, pred= gt[valid.bool()], pred[valid.bool()]

        thresh = np.maximum((gt / pred), (pred / gt))
        a1 = (thresh < 1.25).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        abs_rel = (np.abs(gt - pred) / gt).mean()
        sq_rel =(((gt - pred) ** 2) / gt).mean()

        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        rmse_log = (np.log(gt) - np.log(pred)) ** 2
        rmse_log = np.sqrt(rmse_log.mean())

        err = np.log(pred) - np.log(gt)
        silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

        log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
        
        a1_arr.append(a1)
        a2_arr.append(a2)
        a3_arr.append(a3)
        abs_rel_arr.append(abs_rel)
        rmse_arr.append(rmse)
        log_10_arr.append(log_10)
        rmse_log_arr.append(rmse_log)
        silog_arr.append(silog)
        sq_rel_arr.append(sq_rel)

    a1_arr_mean = sum(a1_arr) / len(a1_arr)
    a2_arr_mean = sum(a2_arr) / len(a2_arr)
    a3_arr_mean = sum(a3_arr) / len(a3_arr)
    abs_rel_arr_mean = sum(abs_rel_arr) / len(abs_rel_arr)
    rmse_arr_mean = sum(rmse_arr) / len(rmse_arr)
    log_10_arr_mean = sum(log_10_arr) / len(log_10_arr)
    rmse_log_arr_mean = sum(rmse_log_arr) / len(rmse_log_arr)
    silog_arr_mean = sum(silog_arr) / len(silog_arr)
    sq_rel_arr_mean = sum(sq_rel_arr) / len(sq_rel_arr)


    return dict(a1=a1_arr_mean, a2=a2_arr_mean, a3=a3_arr_mean, abs_rel=abs_rel_arr_mean, rmse=rmse_arr_mean, log_10=log_10_arr_mean, rmse_log=rmse_log_arr_mean,
                silog=silog_arr_mean, sq_rel=sq_rel_arr_mean)
    #return dict(a1=a1_arr_mean, a2=a2_arr_mean, a3=a3_arr_mean, abs_rel=abs_rel_arr_mean, rmse=rmse_arr_mean, sq_rel=sq_rel_arr_mean)




def calc_metric_avg(metrics_arr):
    len_arr = len(metrics_arr)
    metric_avg = {'a1': 0.0, 'a2': 0.0, 'a3': 0.0, 'abs_rel': 0.0, 'rmse': 3.0, 'log_10': 0.0, 'rmse_log': 0.0, 'silog': 0.0, 'sq_rel': 0.0}
    for metric in metrics_arr:
        for key in metric_avg.keys():
            metric_avg[key] = metric_avg[key] + metric[key]

    for key in metric_avg.keys():
        metric_avg[key] = metric_avg[key]/len_arr
    
    return metric_avg



@torch.no_grad()
def validate_kitti(model, seg_any_predictor: SamPredictor, mixed_prec=False):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.KITTI(seg_any_predictor,aug_params, image_set='training')
    torch.backends.cudnn.benchmark = True

    metrics_arr = []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        #padder = InputPadder(image1.shape, divis_by=32)
        #image1, image2 = padder.pad(image1, image2)

        with torch.cuda.amp.autocast(enabled=mixed_prec):
            start = time.time()
            flow_pr = model(image1, image2)
            end = time.time()

        #flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
        if flow_pr.shape[-2:] != flow_gt.shape[-2:]:
            flow_pr = nn.functional.interpolate(
                flow_pr, size=flow_gt.shape[-2:], mode="bilinear", align_corners=True
            )
        flow_gt = flow_gt.unsqueeze(0)
        valid_gt = valid_gt.unsqueeze(0)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)


        flow_pr = flow_pr * 80
        # flow_pr = ((flow_pr - flow_pr.min()) / (flow_pr.max() - flow_pr.min())) * 255
        # flow_gt = ((flow_gt - flow_gt.min()) / (flow_gt.max() - flow_gt.min())) * 255
        metrics = compute_errors(flow_gt, flow_pr,valid_gt)
        metrics_arr.append(metrics)
        if val_id < 9 or (val_id+1)%10 == 0:
            print(f"KITTI Iter {val_id+1} out of {len(val_dataset)}. {metrics}")


        '''Inference 결과 저장 코드'''
        # outdir = './AsymKD_inference_result'
        # flow_pr = flow_pr.squeeze()
        # flow_pr = (flow_pr - flow_pr.min()) / (flow_pr.max() - flow_pr.min()) * 255.0
        # flow_pr = flow_pr.cpu().numpy().astype(np.uint8)
        # flow_pr = cv2.applyColorMap(flow_pr, cv2.COLORMAP_INFERNO)
        # cv2.imwrite(os.path.join(outdir, 'AsymKD_Feas_'+str(val_id) + '_depth.png'), flow_pr)
        
        
    return calc_metric_avg(metrics_arr)



@torch.no_grad()
def validate_kitti_for_depth_anything(model, seg_any_predictor: SamPredictor, mixed_prec=False):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    aug_params = {}
    val_dataset = datasets.KITTI(seg_any_predictor,aug_params, image_set='training')
    torch.backends.cudnn.benchmark = True

    metrics_arr = []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        #padder = InputPadder(image1.shape, divis_by=32)
        #image1, image2 = padder.pad(image1, image2)

        with torch.cuda.amp.autocast(enabled=mixed_prec):
            start = time.time()
            flow_pr = model(image1)
            end = time.time()

        #flow_pr = padder.unpad(flow_pr).cpu().squeeze(0)
        if flow_pr.shape[-2:] != flow_gt.shape[-2:]:
            flow_pr = nn.functional.interpolate(
                flow_pr, size=flow_gt.shape[-2:], mode="bilinear", align_corners=True
            )
        flow_gt = flow_gt.unsqueeze(0)
        valid_gt = valid_gt.unsqueeze(0)
        print(flow_gt.shape,flow_pr.shape, valid_gt.shape)
        assert flow_pr.shape == flow_gt.shape, (flow_pr.shape, flow_gt.shape)
        #print(flow_pr.min(),flow_pr.max())
        #flow_pr = ((flow_pr - flow_pr.min()) / (flow_pr.max() - flow_pr.min())) * 80
        # flow_gt = ((flow_gt - flow_gt.min()) / (flow_gt.max() - flow_gt.min())) * 255
        metrics = compute_errors(flow_gt, flow_pr,valid_gt)
        metrics_arr.append(metrics)
        if val_id < 9 or (val_id+1)%10 == 0:
            print(f"KITTI Iter {val_id+1} out of {len(val_dataset)}. {metrics}")
        # if val_id < 9 or (val_id+1)%10 == 0:
        #     logging.info(f"KITTI Iter {val_id+1} out of {len(val_dataset)}. {metrics}")

        # '''Inference 결과 저장 코드'''
        # outdir = './Depth_Anything_inference_result'
        # flow_pr = flow_pr.squeeze()
        # flow_pr = (flow_pr - flow_pr.min()) / (flow_pr.max() - flow_pr.min()) * 255.0
        # flow_pr = flow_pr.cpu().numpy().astype(np.uint8)
        # flow_pr = cv2.applyColorMap(flow_pr, cv2.COLORMAP_INFERNO)
        # cv2.imwrite(os.path.join(outdir, 'Depth_Anything_'+str(val_id) + '_depth.png'), flow_pr)

        # '''Inference ground truth 저장 코드'''
        # flow_gt = flow_gt.squeeze()
        # flow_gt = (flow_gt - flow_gt.min()) / (flow_gt.max() - flow_gt.min()) * 255.0
        # flow_gt = flow_gt.cpu().numpy().astype(np.uint8)
        # flow_gt = cv2.applyColorMap(flow_gt, cv2.COLORMAP_INFERNO)
        # cv2.imwrite(os.path.join(outdir, 'Depth_Anything_'+str(val_id) + '_depth_gt.png'), flow_gt)
        
    return calc_metric_avg(metrics_arr)



if __name__ == '__main__':
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = "sam_vit_b_01ec64.pth"
    #checkpoint = "sam_vit_l_0b3195.pth"
    model_type = "vit_b"
    segment_anything = sam_model_registry[model_type](checkpoint=checkpoint).to(DEVICE).eval()
    segment_anything_predictor = SamPredictor(segment_anything)

    '''Depth Anything model load'''
    encoder = 'vitb' # can also be 'vitb' or 'vitl'
    model = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoder)).to(DEVICE).eval()
    results = validate_kitti_for_depth_anything(model,segment_anything_predictor)
    print(f'Depth Anything evaluate result : {results}')    
    print(f'#######Depth Anything evaluate result#############')    
    for key in results.keys():
        print(f'{key} : {round(results[key], 3)}')
    
    '''AsymKD model load'''
    # for child in segment_anything.children():
    #         ImageEncoderViT = child
    #         break
    # model = nn.DataParallel(AsymKD_DepthAnything(ImageEncoderViT = ImageEncoderViT))
    # restore_ckpt = 'checkpoints/160000_AsymKD.pth'
    # #restore_ckpt = 'checkpoints/149981_epoch_AsymKD.pth'
    # if restore_ckpt is not None:
    #         assert restore_ckpt.endswith(".pth")
    #         print("Loading checkpoint...")
    #         checkpoint = torch.load(restore_ckpt)
    #         model.load_state_dict(checkpoint, strict=True)
    #         print(f"Done loading checkpoint")
    # model.to(DEVICE)
    # model.eval()
    # AsymKD_metric = validate_kitti(model,segment_anything_predictor)
    # print(f'AsymKD {restore_ckpt} evaluate result : {AsymKD_metric}')    

    # Depth_Any_metric = {'a1': 0.75005361982926, 'a2': 0.9294285799367217, 'a3': 0.9713775714947123, 'abs_rel': 0.16600794681347908, 'rmse': 7.895601497292518, 'log_10': 0.07704228786285966, 'rmse_log': 0.2627473225072026, 'silog': 21.697275741432478, 'sq_rel': 1.724362504966557}
    
    # print(f'#######AsymKD {restore_ckpt} evaluate result#############')    
    # for key in AsymKD_metric.keys():
    #     print(f'{key} : {round(AsymKD_metric[key], 3)}')
    #     #print(f'diff {key} : {round(Depth_Any_metric[key]-AsymKD_metric[key], 3)}')
    # print(f'#######AsymKD {restore_ckpt} diff evaluate result#############')  
    # for key in AsymKD_metric.keys():
    #     #print(f'{key} : {round(AsymKD_metric[key], 3)}')
    #     print(f'diff {key} : {round(Depth_Any_metric[key]-AsymKD_metric[key], 3)}')

