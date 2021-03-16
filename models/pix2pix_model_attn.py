import torch
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import model_three_loss

class Pix2PixModel(BaseModel):
    def name(self):
        return 'Pix2PixModelThree'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'G_L1', 'G_PL', 'G_SSIM', 'G_MSE', 'D_real', 'D_fake']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A1', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']
        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc*2 + opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD0 = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD1 = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.fake_A0B_pool = ImagePool(opt.pool_size)
            self.fake_A1B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionSL1 = torch.nn.SmoothL1Loss()
            self.criterionVGG = model_three_loss.VGGLoss(self.gpu_ids)
            self.criterionPL = model_three_loss.PerceptualLosses(self.gpu_ids)
            self.criterionSSIM = model_three_loss.SSIM()
            self.criterionMSSSIM = model_three_loss.MSSSIM()
            self.criterionMSE = torch.nn.MSELoss()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=0.0)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=0.0)
            self.optimizer_D0 = torch.optim.Adam(self.netD0.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=0.0)
            self.optimizer_D1 = torch.optim.Adam(self.netD1.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=0.0)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_D0)
            self.optimizers.append(self.optimizer_D1)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.which_epoch)

        self.print_networks(opt.verbose)

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A0 = input['A0' if AtoB else 'B']
        input_A1 = input['A1']
        input_B = input['B' if AtoB else 'A0']
        input_A = torch.cat((input_A0, input_A1), 1)
        if len(self.gpu_ids) > 0:
            input_A0 = input_A0.cuda(self.gpu_ids[0], async=True)
            input_A1 = input_A1.cuda(self.gpu_ids[0], async=True)
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
            input_B = input_B.cuda(self.gpu_ids[0], async=True)
        self.input_A0 = input_A0
        self.input_A1 = input_A1
        self.input_A = input_A
        self.input_B = input_B
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A0 = Variable(self.input_A0)
        self.real_A1 = Variable(self.input_A1)
        self.real_A = Variable(self.input_A)
        self.fake_B = self.netG(self.real_A0, self.real_A1)
        self.real_B = Variable(self.input_B)

    # no backprop gradients
    def test(self):
        self.real_A0 = Variable(self.input_A0, volatile=True)
        self.real_A1 = Variable(self.input_A1, volatile=True)
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG(self.real_A0, self.real_A1)
        self.real_B = Variable(self.input_B, volatile=True)

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        fake_A0B = self.fake_A0B_pool.query(torch.cat((self.real_A0, self.fake_B), 1))
        fake_A1B = self.fake_A1B_pool.query(torch.cat((self.real_A1, self.fake_B), 1))
        pred_fake = self.netD(fake_AB.detach())
        pred_fake0 = self.netD0(fake_A0B.detach())
        pred_fake1 = self.netD1(fake_A1B.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        self.loss_D_fake0 = self.criterionGAN(pred_fake0, False)
        self.loss_D_fake1 = self.criterionGAN(pred_fake1, False)

        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        real_A0B = torch.cat((self.real_A0, self.real_B), 1)
        real_A1B = torch.cat((self.real_A1, self.real_B), 1)
        pred_real = self.netD(real_AB)
        pred_real0 = self.netD0(real_A0B)
        pred_real1 = self.netD1(real_A1B)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        self.loss_D_real0 = self.criterionGAN(pred_real0, True)
        self.loss_D_real1 = self.criterionGAN(pred_real1, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D0 = (self.loss_D_fake0 + self.loss_D_real0) * 0.5
        self.loss_D1 = (self.loss_D_fake1 + self.loss_D_real1) * 0.5

        self.loss_D.backward()
        self.loss_D0.backward()
        self.loss_D1.backward()

    def transform_to_grayscale(self, image):

        # output = image.clone() # makes new memory.
        # output[:, 0, :, :] = 0.299 * image[:, 0, :, :]
        # output[:, 1, :, :] = 0.587 * image[:, 1, :, :]
        # output[:, 2, :, :] = 0.114 * image[:, 2, :, :]

        # output = output.sum(dim=1, keepdim=True)

        output = torch.add(0.299 * image[:, 0, :, :], 1, 0.587 * image[:, 1, :, :])
        output = torch.add(output, 1, 0.113 * image[:, 2, :, :])
        output = torch.unsqueeze(output, dim=1)
        return output

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        fake_A0B = torch.cat((self.real_A0, self.fake_B), 1)
        fake_A1B = torch.cat((self.real_A1, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        pred_fake0 = self.netD0(fake_A0B)
        pred_fake1 = self.netD1(fake_A1B)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True) * 0.001
        self.loss_G_GAN0 = self.criterionGAN(pred_fake0, True) * 0.0001
        self.loss_G_GAN1 = self.criterionGAN(pred_fake1, True)


        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A
        # self.loss_G_SL1 = self.criterionSL1(self.fake_B, self.real_B) * 300
        self.loss_G_VGG = self.criterionVGG(self.fake_B, self.real_B) * 10
        self.loss_G_PL = self.criterionPL(self.fake_B, self.real_B) * 250
        self.loss_G_MSE = self.criterionMSE(self.fake_B, self.real_B) * 1
        self.loss_G_SSIM = (1.0 - self.criterionSSIM(self.transform_to_grayscale(self.fake_B), self.transform_to_grayscale(self.real_B))) * 200
        # self.loss_G_SSIM = (1.0 - self.criterionSSIM(self.real_B, self.fake_B)) * 300
        self.loss_G_MSSSIM = (1.0 - self.criterionMSSSIM(self.transform_to_grayscale(self.real_B), self.transform_to_grayscale(self.fake_B))) * 300
        self.loss_G = self.loss_G_GAN1 \
                    + self.loss_G_L1 \
                    + self.loss_G_PL \
                    #+self.loss_G_GAN +self.loss_G_VGG+ self.loss_G_GAN1 self.loss_G_L1 #self.loss_G_MSE #+ self.loss_G_PL + self.loss_G_SSIM #  #+ self.loss_G_SSIM

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.optimizer_D0.zero_grad()
        self.optimizer_D1.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        self.optimizer_D0.step()
        self.optimizer_D1.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
