from src import method_utils
import torch
import logging


class Inversion(object):

    def __init__(self, args):

        self.args = args
        self.device = torch.device("cuda:0" if self.args.cuda else "cpu")

        self.net = None
        self.dataset = None
        self.trans = None
        self.num_classes = None
        self.mean_std = None

        self.gt_data = None
        self.gt_label = None

        self.original_grad = None

        self._init()
        self.set_logging()

    def _init(self):
        self.net = method_utils.load_net(self.args.net)
        if self.net is not None:
            self.net.to(self.device)
        self.dataset, self.trans, self.num_classes, self.mean_std = method_utils.load_data(self.args.dataset)

        self.get_input_data()

    def set_logging(self):

        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[
                logging.FileHandler('{}/inv.log'.format(self.args.save_path)),
                logging.StreamHandler()
            ]
        )
        logging.info(self.args)

    def get_input_data(self):
        tmp_datum = self.trans(self.dataset[self.args.idx][0]).float().to(self.device)
        self.gt_data = tmp_datum.view(1, *tmp_datum.size())
        tmp_label = torch.Tensor([self.dataset[self.args.idx][1]]).long().to(self.device)
        self.gt_label = tmp_label.view(1, )

    def get_original_grad(self):
        pass

    def inversion(self):
        pass

    def eval(self, dummy_data_path):
        import torchvision

        dummy_data = torch.load(dummy_data_path)
        torchvision.utils.save_image(dummy_data, "res/eval_dummy.png", nrow=1, padding=1)
        torchvision.utils.save_image(self.gt_data, "res/eval_ground_truth.png", nrow=1, padding=1)

        result = method_utils.get_eval(self.args.metrics, dummy_data, self.gt_data)
        logging.info("dataset: {}, idx: {}".format(self.args.dataset, self.args.idx))
        logging.info("mse: {}, lpips: {}, psnr: {}, ssim: {}".format(result[0], result[1], result[2], result[3]))
