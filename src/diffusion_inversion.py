import logging

import torch.optim

from src.inversion_base import *
import torch.nn as nn

from src.method_utils import defense_alg


class DiffusionInversion(Inversion):

    def __init__(self, args, diff_util, prior_util=None, pre_dummy_dir=None):
        super(DiffusionInversion, self).__init__(args)

        self.criterion = nn.CrossEntropyLoss().to(self.device)

        self.net = diff_util
        self.prior = prior_util

        self.t = None
        self.noise = None

        self.pre_dummy_dir = pre_dummy_dir

    def get_original_grad(self):
        method_utils.save_single_img(self.gt_data, "res/running/real_image_idx_{}.png".format(self.args.idx), mean_std=self.mean_std)

        self.t = torch.ones([1]).long().to(self.device)
        self.noise = torch.randn_like(self.gt_data)

        y = self.net.get_loss_t_noise(self.gt_data, self.t, self.noise)

        dy_dx = torch.autograd.grad(y, self.net.diffusion.model.parameters())
        original_grad = list((_.detach().clone() for _ in dy_dx))

        return original_grad

    def get_dummy_image(self, dummy_data):
        if not self.args.using_prior:
            return dummy_data
        return self.prior.sample(num_img=1, input_noise=dummy_data)

    def inversion(self):
        self.original_grad = self.get_original_grad()

        if self.pre_dummy_dir == None:
            dummy_data = torch.randn(self.gt_data.size()).to(self.device).requires_grad_(True)
        else:
            dummy_data = torch.load(self.pre_dummy_dir)
            dummy_data.to(self.device).requires_grad_(True)

        if self.args.known_t:
            dummy_t = self.t
            optimizer_t = None
        else:
            opt_dummy_t = torch.randn((1, 16)).to(self.device).requires_grad_(True)
            optimizer_t = torch.optim.Adam([opt_dummy_t], lr=self.args.lr)

            # dummy_t = torch.randint(0, 16, self.t.shape, device=self.device).long()
            # optimizer_t = None

        dummy_noise = self.noise
        if self.args.known_epsilon:
            optimizer_e = None
        else:
            dummy_noise += torch.randn(self.gt_data.size()).to(self.device) * 0.0001
            optimizer_e = None

        optimizer = torch.optim.Adam([dummy_data], lr=self.args.lr)

        results = []
        for iters in range(self.args.iteration):

            if iters in self.args.save_img_on_iters:
                with torch.no_grad():
                    dummy_img = self.get_dummy_image(dummy_data)
                method_utils.save_single_img(dummy_img, "res/running/dummy_image_idx_{}_iter_{}.png".
                                             format(self.args.idx, iters), mean_std=self.mean_std)
                torch.save(dummy_img, "res/running/s_dummy_image_idx_{}_iter_{}.pth".format(self.args.idx, iters))
                if iters == 0:
                    method_utils.save_single_img(dummy_data, "res/running/noise.png", mean_std=None)
                    torch.save(dummy_data, "res/running/noise.pth")

            if (not self.args.known_t) and (optimizer_t is not None):
                opt_dummy_t = torch.softmax(opt_dummy_t, -1)
                idx = torch.argmax(opt_dummy_t) + 1
                dummy_t = torch.reshape(idx, (1,))

            def closure():
                optimizer.zero_grad()

                dummy_img = self.get_dummy_image(dummy_data)

                dummy_loss = self.net.get_loss_t_noise(dummy_img, dummy_t, dummy_noise)
                dummy_dy_dx = torch.autograd.grad(dummy_loss, self.net.diffusion.model.parameters(), create_graph=True)
                dummy_dy_dx = list((_ for _ in dummy_dy_dx))

                grad_diff = 0
                for c in range(len(dummy_dy_dx)):
                    gx = dummy_dy_dx[c]
                    gy = self.original_grad[c]
                    grad_diff += ((gx - gy) ** 2).sum()

                grad_diff.backward()
                return grad_diff

            def closure_e():
                optimizer_e.zero_grad()

                dummy_img = self.get_dummy_image(dummy_data)
                dummy_loss = self.net.get_loss_t_noise(dummy_img, dummy_t, dummy_noise)
                dummy_dy_dx = torch.autograd.grad(dummy_loss, self.net.diffusion.model.parameters(), create_graph=True)
                dummy_dy_dx = list((_ for _ in dummy_dy_dx))

                grad_diff = 0
                for c in range(len(dummy_dy_dx)):
                    gx = dummy_dy_dx[c]
                    gy = self.original_grad[c]
                    grad_diff += ((gx - gy) ** 2).sum()

                grad_diff.backward()
                return grad_diff

            def closure_t():
                optimizer_t.zero_grad()

                dummy_img = self.get_dummy_image(dummy_data)
                dummy_loss = self.net.get_loss_t_noise(dummy_img, dummy_t, dummy_noise)
                dummy_dy_dx = torch.autograd.grad(dummy_loss, self.net.diffusion.model.parameters(), create_graph=True)
                dummy_dy_dx = list((_ for _ in dummy_dy_dx))

                grad_diff = 0
                for c in range(len(dummy_dy_dx)):
                    gx = dummy_dy_dx[c]
                    gy = self.original_grad[c]
                    grad_diff += ((gx - gy) ** 2).sum()

                grad_diff.backward()
                return grad_diff

            # solution 4
            current_loss = optimizer.step(closure)

            if (optimizer_t is not None) and (iters > 200) and (iters < 210):
                t_loss = optimizer_t.step(closure_t)

            if (int(iters / 10) % 10) == 0 and (optimizer_e is not None) and (iters > 1000) and (iters < 1200):
                e_loss = optimizer_e.step(closure_e)

            if iters % self.args.log_metrics_interval == 0 or iters in [10 * i for i in range(10)]:
                result = method_utils.get_eval(self.args.metrics, dummy_data, self.gt_data)
                logging.info('iters idx: {}, current lr: {}, current loss: {}'.format(iters, optimizer.param_groups[0]['lr'], current_loss))
                # logging.info("real t: {}, dummy t: {}".format(self.t, dummy_t))
                res = [iters]
                res.extend(result)
                results.append(res)

        logging.info("inversion finished")

        return results
