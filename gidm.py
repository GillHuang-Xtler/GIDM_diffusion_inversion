import torchvision
import argparse
import os

from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from src.diffusion_utils import DiffusionUtils

from src.diffusion_inversion import DiffusionInversion


def build_diffusion(image_size, objective="pred_v", timesteps=1024, sampling_timesteps=None, is_student=False, teacher=None, using_ddim=False, use_pdistill=False, mapping_sequence=None):
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        # flash_attn=True
        flash_attn=False
    ).cuda()

    diffusion = GaussianDiffusion(
        model,
        image_size=image_size,  # 32, 64, 128
        timesteps=timesteps,  # number of steps
        sampling_timesteps=sampling_timesteps,
        objective=objective,
        is_student=is_student,
        mapping_sequence=mapping_sequence,
        teacher=teacher,
        using_ddim=using_ddim,
        use_pdistill=use_pdistill
    ).cuda()
    return diffusion


def load_trained_diff(image_size=128, steps=16):
    diff_util = DiffusionUtils(
        build_diffusion(image_size=image_size, timesteps=steps, sampling_timesteps=16, objective="pred_v", using_ddim=True))
    d_dir = 'trained_models/diffusion_celeba_hq_16_128x128_pv_epoch_40.pth'
    diff_util.load_trained_model(d_dir)
    diff_util.diffusion.model.eval()
    return diff_util


def load_untrained_diff(image_size=32, steps=1000):
    diff_util = DiffusionUtils(
        build_diffusion(image_size=image_size, timesteps=steps, sampling_timesteps=steps, objective="pred_v", using_ddim=True))
    diff_util.diffusion.model.eval()
    return diff_util


def get_diff_args(stage=1):
    args = argparse.ArgumentParser()

    args.add_argument('--cuda', default=True, action='store_true', help='using cuda')
    args.add_argument('--dataset', type=str, default='celeba', help='cifar100, celeba, bedroom')
    args.add_argument('--image_size', type=int, default=128)
    args.add_argument('--idx', type=int, default=0)

    args.add_argument('--net', type=str, default='Diffusion', help="LeNet, Diffusion")
    args.add_argument('--defense_method', type=str, default='none', help="none")

    args.add_argument('--known_t', default=False, action='store_true')
    args.add_argument('--known_epsilon', default=False, action='store_true')
    if stage == 1:
        args.add_argument('--using_prior', default=True, action='store_true')
        args.add_argument('--pre_dummy_dir', type=str, default=None)
    else:
        args.add_argument('--using_prior', default=False, action='store_true')
        args.add_argument('--pre_dummy_dir', type=str, default="res/running_stage1/s_dummy_image_idx_0_iter_2500.pth")

    args.add_argument('--iteration', type=int, default=5001)
    args.add_argument('--lr', type=float, default=0.01) # for prior

    args.add_argument('--log_img_num', type=int, default=5)
    args.add_argument('--log_metrics_interval', type=int, default=100)
    args.add_argument('--save_img_on_iters', nargs='+', default=[i*100 for i in range(150)], type=str)

    args.add_argument('--metrics', nargs='+', default=[], type=str)

    args.add_argument('--save_path', type=str, default='res')

    args = args.parse_args()
    return args


def try_diff():
    diff_util = load_trained_diff()
    gen_imgs = diff_util.sample(num_img=16)
    torchvision.utils.save_image(gen_imgs, 'res/visualize.png', nrow=4, padding=2)


def run_diffusion_inv(stage=1, early_stopping=None):
    if not os.path.exists("res"):
        os.makedirs("res")
    if not os.path.exists("res/running"):
        os.makedirs("res/running")

    args = get_diff_args(stage=stage)

    if early_stopping is not None:
        args.iteration = early_stopping

    diff_util = load_untrained_diff(image_size=args.image_size, steps=16)
    if args.using_prior:
        prior_util = load_trained_diff(image_size=args.image_size, steps=16)
        inv = DiffusionInversion(args, diff_util, prior_util=prior_util)
    else:
        inv = DiffusionInversion(args, diff_util, pre_dummy_dir=args.pre_dummy_dir)
    inv.inversion()


if __name__ == '__main__':
    run_diffusion_inv(stage=1, early_stopping=2501)
    os.rename("res/running", "res/running_stage1")
    run_diffusion_inv(stage=2)


