import torch
from torch import nn

from editings.styleclip_mapper import latent_mappers
#from editings.styleclip.model import Generator
from models.stylegan2.model import Generator
from utils.model_utils import setup_model


def load_stylegan_generator(args):
    stylegan_model = Generator(args.stylegan_size, 512, 8, channel_multiplier=2).cuda()
    ckpt = torch.load(args.ckpt)
    # stylegan_model.load_state_dict(checkpoint['g_ema'])
    stylegan_model.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
    return stylegan_model

def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt


class StyleCLIPMapper(nn.Module):

    def __init__(self, opts):
        super(StyleCLIPMapper, self).__init__()
        self.opts = opts
        # Define architecture
        self.mapper = self.set_mapper()
        self.net, _ = setup_model(opts.ckpt, opts.device)
        dict_ = torch.load(opts.ckpt, map_location='cpu')
        self.net.load_state_dict(dict_['state_dict'] , strict=True)        
        self.decoder = load_stylegan_generator(opts)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        # Load weights if needed
        self.load_weights()

    def set_mapper(self):
        if self.opts.mapper_type == 'SingleMapper':
            mapper = latent_mappers.SingleMapper(self.opts)
        elif self.opts.mapper_type == 'LevelsMapper':
            mapper = latent_mappers.LevelsMapper(self.opts)
        else:
            raise Exception('{} is not a valid mapper'.format(self.opts.mapper_type))
        return mapper

    def load_weights(self):
        if self.opts.checkpoint_path is not None:
            print('Loading from checkpoint: {}'.format(self.opts.checkpoint_path))
            ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
            self.mapper.load_state_dict(get_keys(ckpt, 'mapper'), strict=True)

    def forward(self, x, input_code=False):
        if input_code:
            codes = x
        else:
            codes = self.mapper(x)

        return codes