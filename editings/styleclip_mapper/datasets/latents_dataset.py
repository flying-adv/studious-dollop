import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from utils import data_utils
import torchvision
from tqdm import tqdm

transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

def get_dirs(root):
    images = []
    for dirs in tqdm(os.listdir(root)):
        for files in os.listdir(os.path.join(root,dirs)):
            images.append(os.path.join(root,dirs,files))
    return images

class LatentsDataset(Dataset):
    def __init__(self, root, opts, transform=transforms, preprocess=None , return_path=False):
        # self.paths = sorted(data_utils.make_dataset_new(root))
        # self.paths_dir = [os.path.join(root , x) for x in os.listdir(root)[:30]]
        # self.paths = [os.path.join(self.paths_dir,x) for x in os.listdir(self.paths_dir)]
        self.paths = get_dirs(root)
        # print(self.paths)
        self.transform = transform
        self.return_path = return_path
        self.preprocess = preprocess
        self.opts = opts
        

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        from_path = self.paths[index]
        if self.preprocess is not None:
            from_im = self.preprocess(from_path)
        else:
            from_im = Image.open(from_path).convert('RGB')
        if self.transform:
            from_im = self.transform(from_im)

        if self.return_path:
            return from_im , from_path
        else:
            return from_im


# class LatentsDataset(Dataset):

# 	def __init__(self, latents, opts, transforms=None):
# 		self.latents = latents
# 		self.transforms = transforms
# 		self.opts = opts

# 	def __len__(self):
# 		return self.latents.shape[0]

# 	def __getitem__(self, index):
# 		if self.transforms is not None:
# 			return self.latents[index], torch.from_numpy(self.transforms[index][3]).float()
# 		return self.latents[index]