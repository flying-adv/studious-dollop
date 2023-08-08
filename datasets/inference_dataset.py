from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils


class InferenceDataset(Dataset):

    def __init__(self, root, opts, transform=None, preprocess=None):
        self.paths = sorted(data_utils.make_dataset(root))
        self.transform = transform
        self.preprocess = preprocess
        self.opts = opts

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        style_img = '/content/original.jpg'
        from_path = self.paths[index]
        if self.preprocess is not None:
            from_im = self.preprocess(from_path)
            style_img = self.preprocess(style_img)
        else:
            from_im = Image.open(from_path).convert('RGB')
            style_img = Image.open(style_img).convert('RGB')
        if self.transform:
            from_im = self.transform(from_im)
            style_img = self.transform(style_img)
        return from_im,style_img
