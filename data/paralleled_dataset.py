import os.path
import random
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import torch
from .base_dataset import BaseDataset
from .image_folder import make_dataset
from PIL import Image


class ParalleledDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, 'A', opt.phase)
        self.A_path = sorted(make_dataset(self.dir_A))
        self.comps = ['brow', 'cloth', 'eye', 'face', 'hair', 'lip', 'neck', 'nose', 'ear']
        self.dir_B = []
        self.B_paths = []
        for comp in self.comps:
            self.dir_B.append(os.path.join(opt.dataroot+'_'+comp, 'B', opt.phase))
            self.B_paths.append(sorted(make_dataset(self.dir_B[-1])))
 
    def __getitem__(self, index):
        A_path = self.A_path[index]
        B_paths = []
        for path in self.B_paths:
            B_paths.append(path[index])
        A = Image.open(A_path).convert('RGB')
        B = []
        for path in B_paths:
            B.append(Image.open(path).convert('RGB'))
        transforms_ = [
            transforms.Resize((self.opt.fineSize, self.opt.fineSize), InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        t = transforms.Compose(transforms_)
        A = t(A)
        B = list(map(t, B))
        B = torch.cat(B)

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

#        if (not self.opt.no_flip) and random.random() < 0.5:
#            idx = [i for i in range(A.size(2) - 1, -1, -1)]
#            idx = torch.LongTensor(idx)
#            A = A.index_select(2, idx)
#            B = B.index_select(2, idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_paths}

    def __len__(self):
        return len(self.A_path)

    def name(self):
        return 'ParalleledDataset'
