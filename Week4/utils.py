import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import numpy as np
from math import exp


class VOC2007Dataset(Dataset):
    """ 
    
    create Dataset class that takes input of PascalVOC dataset and creates sets of images of 
    sizes X - 72x72x3, y_mid – 144x144x3, y_large – 288x288x3 
    """
    
    def __init__(self, image_set='trainval', root=None, transform=None, sample_slice=None):
        """
        Args:
            image_set (str): one of 'train', 'trainval', or 'val', default "trainval"
            transform (callable, optional): Optional transform to be applied
                on a sample.
            sample_slice (list or tuple): a 2-value list or tuple indicating subset of files to be selected from the dataset
        """
        super(VOC2007Dataset).__init__()
        self.transform = transform
        
        if root is None:
            root = os.path.abspath(os.path.curdir)
        self.root = root
        valid_sets = ['train', 'trainval', 'val']
        assert (image_set in valid_sets), f"{image_set} not among the allowed values. Allowed values are {', '.join(valid_sets)}"
        
        base_dir = os.path.join('VOCdevkit', 'VOC2007')
        voc_root = os.path.join(self.root, base_dir)
        image_dir = os.path.join(voc_root, 'JPEGImages')
        splits_dir = os.path.join(voc_root, 'ImageSets/Main')
        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')
        
        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        
        if sample_slice is None:
            sample_slice = (0, len(file_names))
        assert (sample_slice[-1] <= len(file_names)), f'sample_slice indices are out bounds. Maximum indices: {len(file_names)-1}'
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names[sample_slice[0]:sample_slice[-1]]]
        
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """
        img = cv2.imread(self.images[index], cv2.IMREAD_COLOR)
        img_large = cv2.resize(img, (288, 288))
        img_mid = cv2.resize(img, (144, 144))
        img_small = cv2.resize(img, (72, 72))

        if self.transform is not None:
            img_large = self.transform(img_large)
            img_mid = self.transform(img_mid)
            img_small = self.transform(img_small)

        return dict(y_large=img_large, y_mid=img_mid, X=img_small)
    
    
    def __len__(self):
        return len(self.images)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)