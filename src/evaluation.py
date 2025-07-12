import numpy as np
from math import log10, sqrt
import pytorch_ssim
import torch
import lpips
from torch.autograd import Variable


def PSNR(dummy_data, gt_data):
    '''
    PSNR metric
    '''
    mse = torch.mean((dummy_data - gt_data) ** 2).item()
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def SSIM(dummy_data, gt_data):
    '''
    SSIM metric
    '''
    return pytorch_ssim.ssim(dummy_data, gt_data)


def LPIPS(dummy_data, gt_data):
    '''
    LPIPS metric
    '''
    # loss_fn_alex = lpips.LPIPS(net='alex').cuda()  # best forward scores

    loss_fn_alex = lpips.LPIPS(net='squeeze').cuda()  # best forward scores

    return loss_fn_alex(dummy_data, gt_data)


