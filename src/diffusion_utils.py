import torch
import torchvision


class DiffusionUtils(object):

    def __init__(self, diffusion, student_diff=None, data_loader=None, autoencoder=None):
        self.diffusion = diffusion
        self.data_loader = data_loader
        self.trained_model_dir = None

        self.trained_student_dir = None

        self.student_diff = student_diff

        self.autoencoder = autoencoder

    def set_dataloader(self, data_loader, mean_std=None):
        self.data_loader = data_loader
        self.mean_std = mean_std

    def load_trained_model(self, trained_model_dir):
        self.trained_model_dir = trained_model_dir
        self.diffusion.load_state_dict(torch.load(self.trained_model_dir))
        self.diffusion.eval()

    def get_loss_t_noise(self, images, t, noise):
        loss = self.diffusion.loss_by_t_noise(images, t, noise)
        return loss

    def train(self, epochs=1, start_epochs=0, model_name=""):

        optimizer = torch.optim.Adam(self.diffusion.parameters(), lr=0.0002)

        for e in range(start_epochs, epochs):
            print('epoch: ', e + 1, ' / ', epochs)
            i = 0
            for training_images, y in self.data_loader:

                optimizer.zero_grad()

                loss = self.diffusion(training_images.cuda())

                loss.backward()

                optimizer.step()

                i += 1
                if i % 100 == 0:
                    print('iter: ', i, " / ", len(self.data_loader), " loss: ", loss.item())

            torch.save(self.diffusion.state_dict(), './saved_models/diffusion_{}_epoch_{}.pth'.format(model_name, e+1))
            self.sample(res_id=e + 1)

    def sample(self, res_id=0, num_img=8, nrow=4, save_dir='./res/res_{}.jpg', use_student=False, input_noise=None):
        sampled_images = self.diffusion.sample(batch_size=num_img, input_noise=input_noise)
        # torchvision.utils.save_image(sampled_images, save_dir.format(res_id), nrow=nrow, padding=2)
        return sampled_images

