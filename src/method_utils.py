import torch
import torchvision
import numpy as np
import math
from torchvision import datasets, transforms

from torch.nn.utils import clip_grad_norm_

from src.evaluation import PSNR, SSIM, LPIPS
import matplotlib.pyplot as plt

import conf


def defense_alg(net, gt_data, gt_label, criterion, device, args):
    if args.defense_method == 'soteria':
        return soteria_defense(net, gt_data, gt_label, criterion, device)
    elif args.defense_method == 'noise':
        return noise_defense(net, gt_data, gt_label, criterion, args)
    elif args.defense_method == 'clipping':
        return clipping_defense(net, gt_data, gt_label, criterion, args)
    elif args.defense_method == 'sparsification':
        return sparsification_defense(net, gt_data, gt_label, criterion, device, args)
    else:
        return None


def sparsification_defense(net, gt_data, gt_label, criterion, device, args):
    print("Using sparsification defense")
    out = net(gt_data)
    y = criterion(out, gt_label)
    dy_dx = torch.autograd.grad(y, net.parameters())
    original_dy_dx = list((_.detach().clone() for _ in dy_dx))

    percentage = 100 - args.sparsification_defense_sparsity

    gradient = [None] * len(original_dy_dx)
    for i in range(len(original_dy_dx)):
        grad_tensor = original_dy_dx[i].clone().cpu().detach().numpy()
        flattened_weights = np.abs(grad_tensor.flatten())
        thresh = np.percentile(flattened_weights, percentage)  # 取百分位数
        grad_tensor = torch.where(abs(original_dy_dx[i]) < thresh, 0, original_dy_dx[i])  # 用阈值取梯度
        gradient[i] = torch.Tensor(grad_tensor).to(device)

    return gradient


def noise_defense(net, gt_data, gt_label, criterion, args):
    print("Using noise defense")
    out = net(gt_data)
    y = criterion(out, gt_label)
    dy_dx = torch.autograd.grad(y, net.parameters())
    original_dy_dx = list((_.detach().clone() for _ in dy_dx))

    for j in range(len(original_dy_dx)):
        original_dy_dx[j] += torch.normal(mean=0.0, std=args.noise_std,
                                          size=(original_dy_dx[j]).shape).cuda()

    return original_dy_dx


def clipping_defense(net, gt_data, gt_label, criterion, args):
    print("Using clipping defense")
    out = net(gt_data)
    y = criterion(out, gt_label)
    dy_dx = torch.autograd.grad(y, net.parameters())
    original_dy_dx = list((_.detach().clone() for _ in dy_dx))
    clip_grad_norm_(original_dy_dx, max_norm=args.max_grad_norm_clipping)
    return original_dy_dx

def soteria_defense(net, gt_data, gt_label, criterion, device):
    print("Using soteria defense")

    gt_data.requires_grad = True

    out, feature_fc1_graph = net.forward_with_feature(gt_data)

    deviation_f1_target = torch.zeros_like(feature_fc1_graph)
    deviation_f1_x_norm = torch.zeros_like(feature_fc1_graph)
    for f in range(deviation_f1_x_norm.size(1)):
        deviation_f1_target[:, f] = 1
        feature_fc1_graph.backward(deviation_f1_target, retain_graph=True)
        deviation_f1_x = gt_data.grad.data
        deviation_f1_x_norm[:, f] = (torch.norm(deviation_f1_x.view(deviation_f1_x.size(0), -1), dim=1) /
                                     (feature_fc1_graph.data[:, f]))
        net.zero_grad()
        gt_data.grad.data.zero_()
        deviation_f1_target[:, f] = 0

    # prune r_i corresponding to smallest ||dr_i/dX||/||r_i||
    deviation_f1_x_norm_sum = deviation_f1_x_norm.sum(axis=0)
    thresh = np.percentile(deviation_f1_x_norm_sum.flatten().cpu().numpy(), 1)
    mask = np.where(abs(deviation_f1_x_norm_sum.cpu()) < thresh, 0, 1).astype(np.float32)

    y = criterion(out, gt_label)
    dy_dx = torch.autograd.grad(y, net.parameters())

    original_dy_dx = list((_.detach().clone() for _ in dy_dx))
    # comment this line if you want to try other defense baselines

    original_dy_dx[-2] = original_dy_dx[-2] * torch.Tensor(mask).to(device)
    return original_dy_dx

def gradient_closure2(optimizer, dummy_data, original_dy_dxs, label_preds, nets, args, criterion):
    def TV(x):
        """Anisotropic TV."""
        dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
        dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
        return dx + dy

    def closure():
        optimizer.zero_grad()

        dummy_dy_dxs = []
        for i in range(args.num_servers):
            nets[i].zero_grad()
            pred = (nets[i](dummy_data))
            dummy_loss = criterion(pred, label_preds[i])
            dummy_dy_dxs.append(torch.autograd.grad(dummy_loss, nets[i].parameters(), create_graph=True))

        rec_loss = reconstruction_costs([dummy_dy_dxs[0]], original_dy_dxs[0],
                                        cost_fn='sim', indices='def',
                                        weights='preserve_linear-50',
                                        ignore_zeros=False)

        for i in range(1, args.num_servers):
            rec_loss += reconstruction_costs([dummy_dy_dxs[i]], original_dy_dxs[i],
                                             cost_fn='sim', indices='def',
                                             weights='preserve_linear-50',
                                             ignore_zeros=False)

        rec_loss += 0.0001 * TV(dummy_data)
        rec_loss.backward()
        return rec_loss

    return closure


def gradient_closure(optimizer, dummy_data, original_dy_dxs, label_preds, nets, args, criterion):
    def TV(x):
        """Anisotropic TV."""
        dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
        dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
        return dx + dy

    def closure():
        optimizer.zero_grad()

        dummy_dy_dxs = []
        for i in range(args.num_servers):
            nets[i].zero_grad()
            pred = (nets[i](dummy_data))
            dummy_loss = criterion(pred, label_preds[i])
            dummy_dy_dxs.append(torch.autograd.grad(dummy_loss, nets[i].parameters(), create_graph=True))

        grad_diff = 0

        # for i in range(args.num_servers):
        #     for gx, gy in zip(dummy_dy_dxs[i], original_dy_dxs[i]):
        #         grad_diff += ((gx - gy) ** 2).sum()
        # grad_diff.backward()

        for j in torch.arange(len(original_dy_dxs)):
            pnorm = [0, 0]
            costs = 0
            for gx, gy in zip(dummy_dy_dxs[j], original_dy_dxs[j]):
                costs -= (gx * gy).sum()
                pnorm[0] += gx.pow(2).sum()
                pnorm[1] += gy.pow(2).sum()
            costs = 1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()
            grad_diff += costs
        grad_diff = grad_diff / len(original_dy_dxs)

        grad_diff += 0.0001 * TV(dummy_data)

        grad_diff.backward()

        return grad_diff

    return closure

def reconstruction_costs(gradients, input_gradient, cost_fn='l2', indices='def', weights='equal', ignore_zeros=False):
    """Input gradient is given data."""
    if isinstance(indices, list):
        pass
    elif isinstance(indices, tuple) and len(indices) == 2:
        indices = torch.arange(indices[0], len(input_gradient), indices[1])
    elif indices == 'def' or indices == 'interleave' or indices == 'concat':
        indices = torch.arange(len(input_gradient))
    elif indices == 'batch':
        indices = torch.randperm(len(input_gradient))[:8]
    elif indices == 'topk-1':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 4)
    elif indices == 'top10':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 10)
    elif indices == 'top50':
        _, indices = torch.topk(torch.stack([p.norm() for p in input_gradient], dim=0), 50)
    elif indices in ['first', 'first4']:
        indices = torch.arange(0, 4)
    elif indices == 'first5':
        indices = torch.arange(0, 5)
    elif indices == 'first10':
        indices = torch.arange(0, 10)
    elif indices == 'first50':
        indices = torch.arange(0, 50)
    elif indices == 'last5':
        indices = torch.arange(len(input_gradient))[-5:]
    elif indices == 'last10':
        indices = torch.arange(len(input_gradient))[-10:]
    elif indices == 'last50':
        indices = torch.arange(len(input_gradient))[-50:]
    elif indices == 'half':
        n = len(input_gradient)
        indices = torch.cat([torch.arange(0, n - 2, 6), torch.arange(1, n - 2, 6), torch.arange(2, n - 2, 6), torch.arange(n - 2, n)])
    else:
        raise ValueError()

    ex = input_gradient[0]
    global_weights = None
    global_weights_flag = False

    if type(weights) is str and weights.startswith('preserve_'):
        preserve_flag = True
        weights = weights.replace("preserve_", "")
    else:
        preserve_flag = False
    if type(weights) is str and '-' in weights:
        scale = float(weights.split('-')[-1])
        weights = weights.split('-')[0]
    else:
        scale = 20

    if global_weights_flag and global_weights is not None:
        weights = global_weights
    elif torch.is_tensor(weights) and weights.requires_grad:
        cnn_mean = torch.mean(weights[:-2])
        with torch.no_grad():
            weights[-1] = weights[-2] = cnn_mean
        weights = torch.softmax(weights, dim=0)
        print(weights)
    elif type(weights) != str:
        pass
    elif weights == 'rev_linear':
        weights = torch.arange(len(input_gradient), 0, -1, dtype=ex.dtype, device=ex.device) / len(input_gradient)
        weights = reversed(weights)
    elif weights == 'power':
        power = scale
        weights = input_gradient[0].new_ones(len(input_gradient))
        n = len(weights)
        for k in range(1, n // 3):
            weights[3 * k] = weights[3 * k + 1] = weights[3 * k + 2] = (k + 1) ** power
        weights[-1] = weights[-2] = torch.mean(weights[:-2])
    elif weights == 'l2linear' or weights == 'linear':
        weights = input_gradient[0].new_ones(len(input_gradient))
        n = len(weights)
        for k in range(1, n // 3):
            weights[3 * k] = weights[3 * k + 1] = weights[3 * k + 2] = 1 + (scale - 1) / (n // 3 - 1) * k
        weights[-1] = weights[-2] = torch.mean(weights[:-2])
    elif weights == '2part':
        weights = input_gradient[0].new_ones(len(input_gradient))
        for k, w in enumerate(input_gradient):
            weights[k] = 1 / torch.mean(torch.sum(w ** 2))
        weights /= sum(weights)
    elif weights == 'log':
        weights = input_gradient[0].new_ones(len(input_gradient))
        n = len(weights)
        for k in range(1, n // 3):
            weights[3 * k] = weights[3 * k + 1] = weights[3 * k + 2] = 1 + math.log(k + 1) / math.log(n // 3) * (scale - 1)
        weights[-1] = weights[-2] = torch.mean(weights[:-2])
    elif weights == 'quad':
        weights = input_gradient[0].new_ones(len(input_gradient))
        n = len(weights)
        for k in range(1, n // 3):
            weights[3 * k] = weights[3 * k + 1] = weights[3 * k + 2] = 1 + math.pow(k + 1, 2) / math.pow(n // 3, 2) * (scale - 1)
        weights[-1] = weights[-2] = torch.mean(weights[:-2])
    elif weights == 'inv':
        weights = torch.arange(len(input_gradient), 0, -1, dtype=ex.dtype, device=ex.device) / len(input_gradient)
        weights = 1 / weights
    elif weights == 'exp':
        weights = torch.arange(len(input_gradient), 0, -1, dtype=ex.dtype, device=ex.device)
        weights = weights.softmax(dim=0)
        weights = weights / weights[0]
    elif weights == 'ratio':
        ratio = math.e
        weights = input_gradient[0].new_ones(len(input_gradient))
        n = len(weights)
        for k in range(1, n // 3):
            weights[3 * k] = weights[3 * k + 1] = weights[3 * k + 2] = 1 + math.pow(ratio, k) / math.pow(ratio, n // 3 - 1) * (scale - 1)
        weights[-1] = weights[-2] = torch.mean(weights[:-2])
    elif weights == 'mean':
        weights = input_gradient[0].new_ones(len(input_gradient))
        for i in range(len(weights)):
            weights[i] = 1 / torch.mean(torch.abs(input_gradient[i])).item()
    elif weights == 'var':
        weights = input_gradient[0].new_ones(len(input_gradient))
        for i in range(len(weights)):
            weights[i] = 1 / torch.var(input_gradient[i]).item()
    elif weights == 'linearvar':
        weights = input_gradient[0].new_ones(len(input_gradient))
        for i in range(len(weights)):
            weights[i] = i / torch.var(input_gradient[i]).item()
    elif weights == 'std':
        weights = input_gradient[0].new_ones(len(input_gradient))
        for i in range(len(weights)):
            weights[i] = 1 / torch.std(input_gradient[i]).item()
    elif weights == 'rev_exp':
        weights = torch.arange(len(input_gradient), 0, -1, dtype=ex.dtype, device=ex.device)
        weights = weights.softmax(dim=0)
        weights = weights / weights[0]
        weights = reversed(weights)
    elif weights == 'preserve':
        weights = [1] * len(input_gradient)
        for i in range(len(weights)):
            preserve_rate = (max(1, torch.count_nonzero(input_gradient[i]).item()) / np.prod(input_gradient[i].size()))
            weights[i] = 1 / preserve_rate
    elif weights == 'concave':
        weights = input_gradient[0].new_ones(len(input_gradient))
        n = len(weights)
        for k in range(1, n // 3):
            weights[3 * k] = weights[3 * k + 1] = weights[3 * k + 2] = 1 + math.log(k + 1)
        weights[:-2] = torch.min(weights[:-2], weights[:-2].__reversed__())
        weights = 1 + (weights - 1) * (scale - 1) / (torch.max(weights) - 1)
        weights[-1] = weights[-2] = torch.mean(weights[:-2])
    elif weights == 'equal':
        weights = input_gradient[0].new_ones(len(input_gradient))
    else:
        raise NotImplementedError

    if preserve_flag and not global_weights_flag:
        for i in range(len(weights) - 2):
            preserve_rate = (max(1, torch.count_nonzero(input_gradient[i]).item()) / np.prod(input_gradient[i].size()))
            weights[i] = weights[i] / preserve_rate

    if not global_weights_flag:
        global_weights_flag = True
        global_weights = weights

    total_costs = 0
    for trial_gradient in gradients:
        trial_gradient = list(trial_gradient)
        pnorm = [0, 0]
        costs = 0
        tmp = 0
        if indices == 'topk-2':
            _, indices = torch.topk(torch.stack([p.norm().detach() for p in trial_gradient], dim=0), 4)
        for i in indices:
            if ignore_zeros:
                trial_gradient[i] = torch.where(input_gradient[i] == 0, input_gradient[i], trial_gradient[i])
            if cost_fn == 'l2':
                costs += ((trial_gradient[i] - input_gradient[i]).pow(2)).sum() * weights[i]
            elif cost_fn == 'l1':
                costs += ((trial_gradient[i] - input_gradient[i]).abs()).sum() * weights[i]
            elif cost_fn == 'max':
                costs += ((trial_gradient[i] - input_gradient[i]).abs()).max() * weights[i]
            elif cost_fn == 'sim':
                costs -= (trial_gradient[i] * input_gradient[i]).sum() * weights[i]
                pnorm[0] += trial_gradient[i].pow(2).sum() * weights[i]
                pnorm[1] += input_gradient[i].pow(2).sum() * weights[i]
            elif cost_fn == 'sum':
                costs -= (trial_gradient[i] * input_gradient[i]).sum() * weights[i]
                pnorm[0] += trial_gradient[i].pow(2).sum() * weights[i]
                pnorm[1] += input_gradient[i].pow(2).sum() * weights[i]
                tmp += ((trial_gradient[i] - input_gradient[i]).pow(2)).sum() * weights[i]
            elif cost_fn == 'simlocal':
                costs += 1 - torch.nn.functional.cosine_similarity(trial_gradient[i].flatten(),
                                                                   input_gradient[i].flatten(),
                                                                   0, 1e-10) * weights[i]
            elif cost_fn == 'gaussian':
                costs += 1 - torch.exp(- ((trial_gradient[i] - input_gradient[i]).pow(2)).sum() /
                                       torch.var(input_gradient[i]))
        if cost_fn == 'sim':
            costs = 1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()
        if cost_fn == 'sum':
            costs = 1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt() + tmp
        # Accumulate final costs
        total_costs += costs
    return total_costs / len(gradients)


def load_net(net_name):
    net = None
    if net_name == "LeNet":
        from src.lenet import LeNet
        net = LeNet(channel=3, hidden=768, num_classes=100)
        net.eval()
    elif net_name == 'Diffusion':
        pass
    return net


def load_data(dataset, data_path=None):
    if data_path is None:
        import conf
        data_path = conf.data_path

    d_mean = [0.5, 0.5, 0.5]
    d_std = [0.5, 0.5, 0.5]
    num_classes = None

    if dataset == 'cifar100':
        dst = datasets.CIFAR100(data_path, download=True)
        d_mean = [0.5071598291397095, 0.4866936206817627, 0.44120192527770996]
        d_std = [0.2673342823982239, 0.2564384639263153, 0.2761504650115967]
        mean_std = (d_mean, d_std)
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(d_mean, d_std)])
        num_classes = 100
    elif dataset == 'celeba':
        data_path = conf.data_path + '/celeba_hq_256/'
        dst = torchvision.datasets.ImageFolder(root=data_path)
        d_mean = [0.5071598291397095, 0.4866936206817627, 0.44120192527770996]
        d_std = [0.2673342823982239, 0.2564384639263153, 0.2761504650115967]
        mean_std = (d_mean, d_std)
        mean_std = None
        trans = transforms.Compose([
            transforms.CenterCrop((256, 256)),
            transforms.Resize([128, 128]),
            transforms.ToTensor(),
            # transforms.Normalize(d_mean, d_std),
        ])
    elif dataset == 'bedroom':
        train_set = datasets.LSUN(root=conf.data_path, classes=['bedroom_train'])
        mean_std = None
        trans = transforms.Compose([
            transforms.CenterCrop((256, 256)),
            transforms.Resize([128, 128]),
            transforms.ToTensor(),
        ])

        subset_size = int(len(train_set) / 200)
        subset_indices = list(range(0, subset_size))
        dst = torch.utils.data.Subset(train_set, subset_indices)
    else:
        dst = None
        trans = None
        num_classes = None
        mean_std = None

    return dst, trans, num_classes, mean_std


def get_eval(eval_metrics, dummy_data, gt_data):
    mse = psnr = ssim = lpips = 0
    if 'mse' in eval_metrics:
        mse = (torch.mean((dummy_data - gt_data) ** 2).item())
    if 'psnr' in eval_metrics:
        psnr = float(PSNR(dummy_data, gt_data))
    if 'ssim' in eval_metrics:
        ssim = float(SSIM(dummy_data, gt_data))
    if 'lpips' in eval_metrics:
        lpips = float(LPIPS(dummy_data, gt_data))

    return [mse, lpips, psnr, ssim]


def save_single_img(data, save_dir, mean_std=None):
    if mean_std is not None:
        d_mean, d_std = mean_std
        dm = torch.as_tensor(d_mean)[:, None, None].cuda()
        ds = torch.as_tensor(d_std)[:, None, None].cuda()
        data = torch.clamp(data * ds + dm, 0, 1)

    torchvision.utils.save_image(data, save_dir, nrow=1, padding=0)


def save_img(iters, history, dummy_data, num_dummy, history_iters, gt_data, mean_std=None):
    tp = transforms.Compose([transforms.ToPILImage()])

    if mean_std is not None:
        d_mean, d_std = mean_std
        dm = torch.as_tensor(d_mean)[:, None, None].cuda()
        ds = torch.as_tensor(d_std)[:, None, None].cuda()
        dummy_data = torch.clamp(dummy_data * ds + dm, 0, 1)
        gt_data = torch.clamp(gt_data * ds + dm, 0, 1)

    history.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
    history_iters.append(iters)

    for imidx in range(num_dummy):
        plt.figure(figsize=(12, 8))
        plt.subplot(1, 7, 1)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.imshow(tp(gt_data[imidx].cpu()))
        plt.title('original')
        plt.axis('off')
        for i in range(min(len(history), 6)):
            plt.subplot(1, 7, i + 2)
            plt.imshow(history[i][imidx])
            plt.title('iter=%d' % (history_iters[i]))
            plt.axis('off')

        plt.savefig('res/res_diff.png')
        plt.close()
