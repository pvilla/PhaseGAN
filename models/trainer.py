import itertools
import os
from torch import nn
import numpy as np
import torchvision.transforms as transforms
from collections import OrderedDict
from abc import ABC
from models.networks import PRNet
from models.prop import Propagator
from models.discriminator import NLayerDiscriminator
from models.initialization import init_weights
from dataset.Dataset2channel import *
import matplotlib.pyplot as plt

class TrainModel(ABC):
    def __init__(self,opt):
        self.opt = opt
        self.load_path = opt.load_path
        self.save_path = opt.run_path
        self.run_name = opt.run_name
        self.batch_size = opt.batch_size
        self.lambda_GA = opt.lambda_GA
        self.lambda_GB = opt.lambda_GB
        self.lambda_A = opt.lambda_A
        self.lambda_B = opt.lambda_B
        self.lambda_fscA = opt.lambda_FSCA
        self.lambda_fscB = opt.lambda_FSCB
        self.pretrained = not opt.no_pretrain
        self.num_epochs = opt.num_epochs
        self.lr_g = opt.lr_g
        self.lr_d = opt.lr_d
        self.beta1 = opt.beta1
        self.clip_max = opt.clip_max
        self.E = opt.energy
        self.pxs = opt.pxs
        self.z = opt.z
        self.wavelength = 12.4/self.E*1e-10
        self.images_mean,self.images_std,self.reals_mean,self.reals_std,self.imags_mean,self.imags_std = opt.image_stats
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('device:{}'.format(self.device))
        self.model_init_type = 'normal'
        self.criterionBSE = nn.BCELoss()
        self.criterionCycle = nn.L1Loss()
        self.isTrain = not opt.isTest
        self.adjust_lr_epoch = opt.adjust_lr_epoch

        self.log_note = opt.log_note
        self.save_run = F"{self.save_path}/{self.run_name}"
        self.save_log = F"{self.save_run}/log.txt"
        self.save_stats = F"{self.save_run}/stats.txt"
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'D_B', 'G_B', 'cycle_B','fscA','fscB']
        self.img_names = ['real_A', 'fake_B', 'prop_A', 'rec_A', 'real_B', 'prop_B', 'fake_A', 'rec_B']

    def get_current_losses(self):
        errors_list = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_list[name] = float(
                    getattr(self, 'loss_' + name))
        return errors_list

    def save_parameters(self):
        with open(self.save_log, "w+") as f:
            f.write(
                "Training log for {}\r\n\n{}\nlr_g:{}\nlr_d:{}\nlambda_GA:{}\nlambda_GB:{}\nlambda_A:{}\nlambda_B:{}\nlambda_fscA:{}\nlambda_fscB:{}\nbeta1:{}\nclip_max:{}\n\n".format(
                    self.run_name, self.log_note, self.lr_g, self.lr_d, self.lambda_GA, self.lambda_GB, self.lambda_A, self.lambda_B, self.lambda_fscA,self.lambda_fscB,self.beta1, self.clip_max))

    def print_current_losses(self, epoch, iters, losses):
        message = 'Epoch [{}/{}], Step [{}/{}]'.format(epoch+1, self.num_epochs, iters+1, self.total_step)
        for name, loss in losses.items():
            message += ', {:s}: {:.3f}'.format(name, loss)
        print(message)
        with open(self.save_log, "a") as f:
            print(message,file=f)

    def adjust_learning_rate(self, epoch, optimizer, initial_lr):
        """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
        lr = initial_lr * (0.1 ** (epoch // self.adjust_lr_epoch))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def update_learning_rate(self,epoch):
        self.adjust_learning_rate(epoch, self.optimizer_G,self.lr_g)
        self.adjust_learning_rate(epoch, self.optimizer_D,self.lr_d)

    def load_data(self):
        print('start loading data....')
        if self.isTrain:
            train_dataset = Dataset2channel(self.load_path, recursive=False, load_data=False,
                                            data_cache_size=1, transform=transforms.ToTensor())
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        else:
            train_loader = []
        test_dataset = Dataset2channel(self.load_path + '/test', recursive=False, load_data=False,
                                       data_cache_size=1, transform=transforms.ToTensor())
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=self.batch_size, shuffle=False)
        print('finish loading data')
        return train_loader, test_loader

    def get_model(self, num_out=1):
        model = PRNet(pretrained=self.pretrained, num_out=num_out)
        model.eval()
        if self.pretrained is not True:
            init_weights(model, self.model_init_type, init_gain=0.02)
        return model

    def get_NLdnet(self, num_input=1):
        dnet = NLayerDiscriminator(num_input)
        dnet.eval()
        init_weights(dnet, self.model_init_type, init_gain=0.02)
        return dnet

    def create_dir_if_not_exist(self, path):
        if os.path.exists(path):
            decision = input('This folder already exists. Continue training will overwrite the data. Proceed(y/n)?')
            if decision != 'y':
                exit()
            else:
                print('Warning: Overwriting folder: {}'.format(self.run_name))
        if not os.path.exists(path):
            os.makedirs(path)

    def init_model(self):
        self.netG_A = nn.DataParallel(self.get_model(num_out=2)).to(self.device)
        self.netG_B = nn.DataParallel(self.get_model(num_out=1)).to(self.device)
        self.netD_A = nn.DataParallel(self.get_NLdnet(2)).to(self.device)
        self.netD_B = nn.DataParallel(self.get_NLdnet()).to(self.device)
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=self.lr_g, betas=(self.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=self.lr_d, betas=(self.beta1, 0.999))
        self.train_loader, self.test_loader = self.load_data()
        self.total_step = len(self.train_loader)
        self.create_dir_if_not_exist(self.save_run)
        self.save_parameters()

    def optimize_parameters(self):
        self.init_model()

    def set_input(self,input):
        self.r_index = torch.randperm(self.batch_size)
        self.images, self.reals,self.imags = [input[i].to(self.device,dtype=torch.float) for i in range(3)]
        self.real_A = (self.images.to(self.device) - self.images_mean)/self.images_std
        self.real_B_re_rc = self.reals[self.r_index][:, :].to(self.device)
        self.real_B_im_rc = self.imags[self.r_index][:, :].to(self.device)
        self.real_B_ph = torch.atan2(self.real_B_im_rc, self.real_B_re_rc).unsqueeze(1)
        self.real_B = self.standard_channels_realB()
        self.real_B_rc = torch.stack((self.real_B_re_rc, self.real_B_im_rc), -1)

    def val_input(self,input):
        self.images, self.reals, self.imags = [input[i].to(self.device, dtype=torch.float) for i in range(3)]
        self.real_A = (self.images.to(self.device) - self.images_mean) / self.images_std
        self.real_B_re_rc = self.reals.to(self.device)
        self.real_B_im_rc = self.imags.to(self.device)
        self.real_B_ph = torch.atan2(self.real_B_im_rc, self.real_B_re_rc).unsqueeze(1)
        self.real_B = self.standard_channels_realB()
        self.real_B_rc = torch.stack((self.real_B_re_rc, self.real_B_im_rc), -1)

    def standard_channels_realB(self):
        real_B_re = (self.real_B_re_rc - self.reals_mean) / self.reals_std
        real_B_im = (self.real_B_im_rc - self.imags_mean) / self.imags_std
        real_B = torch.stack((real_B_re, real_B_im), 1)
        return real_B
    def standard_channelsB_basic(self,layers_B):
        layers_B_re_rc = layers_B[:, 0, :, :] * self.reals_std + self.reals_mean
        layers_B_im_rc = layers_B[:, 1, :, :] * self.imags_std + self.imags_mean
        layers_B_rc = torch.stack((layers_B_re_rc, layers_B_im_rc), -1)
        layers_B_ph = torch.atan2(layers_B_im_rc, layers_B_re_rc).unsqueeze(1)
        return layers_B_ph, layers_B_rc

    def standard_prop_basic(self, img):
        prop_rc = Propagator(self.opt).fresnel_prop(img.squeeze())
        prop = (prop_rc - self.images_mean) / self.images_std
        prop = prop.unsqueeze(1)
        return prop

    def GANLoss(self, pred, is_real):
        target = self.logic_tensor(pred, is_real)
        loss = self.criterionBSE(pred, target)
        return loss

    def FRCLoss(self, img1, img2):
        nz,nx,ny= [torch.tensor(i, device=self.device) for i in img1.shape]
        rnyquist = nx//2
        x = torch.cat((torch.arange(0, nx / 2), torch.arange(-nx / 2, 0))).to(self.device)
        y = x
        X, Y = torch.meshgrid(x, y)
        map = X ** 2 + Y ** 2
        index = torch.round(torch.sqrt(map.float()))
        r = torch.arange(0, rnyquist + 1).to(self.device)
        F1 = torch.rfft(img1, 2, onesided=False).permute(1, 2, 0, 3)
        F2 = torch.rfft(img2, 2, onesided=False).permute(1, 2, 0, 3)
        C_r,C1,C2,C_i = [torch.empty(rnyquist + 1, self.batch_size).to(self.device) for i in range(4)]
        for ii in r:
            auxF1 = F1[torch.where(index == ii)]
            auxF2 = F2[torch.where(index == ii)]
            C_r[ii] = torch.sum(auxF1[:, :, 0] * auxF2[:, :, 0] + auxF1[:, :, 1] * auxF2[:, :, 1], axis=0)
            C_i[ii] = torch.sum(auxF1[:, :, 1] * auxF2[:, :, 0] - auxF1[:, :, 0] * auxF2[:, :, 1], axis=0)
            C1[ii] = torch.sum(auxF1[:, :, 0] ** 2 + auxF1[:, :, 1] ** 2, axis=0)
            C2[ii] = torch.sum(auxF2[:, :, 0] ** 2 + auxF2[:, :, 1] ** 2, axis=0)

        FRC = torch.sqrt(C_r ** 2 + C_i ** 2) / torch.sqrt(C1 * C2)
        FRCm = 1 - torch.where(FRC != FRC, torch.tensor(1.0, device=self.device), FRC)
        My_FRCloss = torch.mean((FRCm) ** 2)
        return My_FRCloss

    def plot_cycle(self, img_idx,save_name, layer=0, test=False, plot_phase=True):
        """set layer to 1 and plot_phase to False to plot imag channel"""
        img_list = ['real_A', 'fake_B', 'prop_A', 'rec_A', 'real_B', 'prop_B', 'fake_A', 'rec_B']
        if plot_phase is True:
            img_list[1], img_list[4], img_list[7] = [img_list[i] + '_ph' for i in [1, 4, 7]]

        fig, axs = plt.subplots(2, int(len(img_list) / 2), figsize=(20, 20), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace=0.0001, wspace=0.0001)
        axs = axs.ravel()
        ignore = [1, 4, 7]  # list of indices to be ignored
        l = [idx for idx in range(len(img_list)) if idx not in ignore]
        for i in l:
            if test == True:
                im = axs[i].imshow(getattr(self,img_list[i])[img_idx, 0, :, :].cpu())
            else:
                im = axs[i].imshow(getattr(self,img_list[i])[img_idx, 0, :, :].detach().cpu())
            axs[i].axis("off")
            axs[i].set_title(img_list[i], fontsize=36)
        for i in ignore:
            if test == True:
                im = axs[i].imshow(getattr(self,img_list[i])[img_idx, layer, :, :].cpu())
            else:
                im = axs[i].imshow(getattr(self,img_list[i])[img_idx, layer, :, :].detach().cpu())
            axs[i].axis("off")
            axs[i].set_title(img_list[i], fontsize=36)
        if save_name != 0:
            if test is True:
                save_path = F"{self.save_run}/test"
            else:
                save_path = F"{self.save_run}/train"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(save_path + F"/{save_name}.png")
            plt.cla()
            plt.close()

    def logic_tensor(self, pred, is_real):
        if is_real:
            target = torch.tensor(1.0)
        else:
            target = torch.tensor(0.0)
        return target.expand_as(pred).to(self.device)

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def backward_D(self, netD, real, fake):
        pred_real = netD(real)
        loss_D_real = self.GANLoss(pred_real, True)
        pred_fake = netD(fake.detach())
        loss_D_fake = self.GANLoss(pred_fake, False)
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_G(self):
        self.loss_G_A = self.GANLoss(self.netD_A(self.fake_B), True) * self.lambda_GA
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * self.lambda_B
        self.loss_G_B = self.GANLoss(self.netD_B(self.fake_A), True) * self.lambda_GB
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * self.lambda_A
        self.loss_fscA = 0
        self.loss_fscB = 0
        if self.lambda_fscA + self.lambda_fscB > 0:
            self.loss_fscA = self.lambda_fscA * self.FRCLoss(self.rec_A.squeeze(), self.real_A.squeeze())
            self.loss_fscB = self.lambda_fscB / 2 * (
                    self.FRCLoss(self.rec_B[:, 0, :, :].squeeze(), self.real_B[:, 0, :, :].squeeze()) + self.FRCLoss(
                self.rec_B[:, 1, :, :].squeeze(),  self.real_B[:, 1, :, :].squeeze()))
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_fscA + self.loss_fscB
        self.loss_G.backward()
        if self.clip_max != 0:
            nn.utils.clip_grad_norm_(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), self.clip_max)

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)
        self.fake_B_ph, self.fake_B_rc = self.standard_channelsB_basic(self.fake_B)
        self.prop_A = self.standard_prop_basic(self.fake_B_rc)
        self.rec_A = self.netG_B(self.prop_A)
        self.prop_B =self.standard_prop_basic(self.real_B_rc)
        self.fake_A = self.netG_B(self.prop_B)
        self.rec_B = self.netG_A(self.fake_A)
        self.rec_B_ph, self.rec_B_rc = self.standard_channelsB_basic(self.rec_B)

    def optimization(self):
        self.forward()
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.loss_D_A = self.backward_D(self.netD_A, self.real_B, self.fake_B)
        self.loss_D_B = self.backward_D(self.netD_B, self.real_A, self.fake_A)
        self.optimizer_D.step()
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def write_to_stat(self, epoch,iter):
        with open(self.save_stats, "a+") as f:
            f.write('\n -------------------------------------------------------\nEpoch [{}/{}], Step [{}/{}]\n'.format(
                epoch + 1, self.num_epochs, iter + 1, self.total_step))
            for i in range(len(self.img_names)):
                self.print_numpy_to_log(getattr(self,self.img_names[i]).detach().cpu().numpy(), f, self.img_names[i])

    def save_net(self, name,epoch,net,optimizer,loss):
        model_save_name = F'{self.run_name}_{name}_{epoch}ep.pt'
        path = F"{self.save_run}/save"
        if not os.path.exists(path):
            os.makedirs(path)
        print('saving trained model {}'.format(model_save_name))
        torch.save({
          'epoch': epoch,
          'model_state_dict': net.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'loss': loss}, path+F'/{model_save_name}')

    def print_numpy_to_log(self, x, f, note):
        x = x.astype(np.float64)
        x = x.flatten()
        print('%s:  mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (note,np.mean(x), np.min(x),np.max(x),np.median(x), np.std(x)),file=f)

    def visual_iter(self,epoch,iter):
        self.write_to_stat(epoch, iter)
        save_name = '{:03d}epoch_{:04d}step'.format(epoch + 1, iter + 1)
        self.plot_cycle(0, save_name)

    def visual_val(self,epoch,idx):
        save_name = '{:03d}epoch_{:02d}'.format(epoch + 1, idx+1)
        self.plot_cycle(0,save_name,0,True)

    def save_models(self, epoch):
        self.save_net('netG_A', epoch + 1, self.netG_A, self.optimizer_G, self.loss_G_A)
        self.save_net('netG_B', epoch + 1, self.netG_B, self.optimizer_G, self.loss_G_B)
        self.save_net('netD_A', epoch + 1, self.netD_A, self.optimizer_D, self.loss_D_A)
        self.save_net('netD_B', epoch + 1, self.netD_B, self.optimizer_D, self.loss_D_B)
