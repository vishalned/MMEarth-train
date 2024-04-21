# a file to help visualizing the masking and output of the model.

import torch
import numpy as np

import matplotlib.pyplot as plt
import models.fcmae as fcmae
from main_pretrain import get_args_parser
from MODALITIES import MODALITIES, MODALITIES_FULL
import os


mean_l2a = [1454.052043293261,
            1698.882623544484,
            1875.142541237718]
std_l2a = [2341.9789148958757,
            2230.0420050925604,
            2327.811137649639]

mean_l1c = [1667.2024400192367,
            1575.6775816668842,
            1654.88549977273]

std_l1c = [1576.4262889011543,
            1480.4841340618689,
            1740.126647255253]



def show_image(img, title='', type=''):
    assert img.shape[2] == 3
    if type == 'l2a':
        mean = np.array(mean_l2a)
        std = np.array(std_l2a)
    elif type == 'l1c':
        mean = np.array(mean_l1c)
        std = np.array(std_l1c)

    # img = torch.from_numpy(img)
    # print(img.shape)
    img = img.permute(2, 0, 1)
    # print(img.shape)
    img = img*std[:, None, None] + mean[:, None, None]
    
    img = img/10000
    clip_val = 0.2
    img = np.clip(img, 0, clip_val)
    img = img / clip_val # clamp to 0-1
    img = torch.einsum('chw->hwc', img)
    # sentinel2 image is in the order of b2, b3, b4. We need to change it to b4, b3, b2 since that is the order of RGB
    img = img[:, :, [2, 1, 0]]
    plt.imshow(img)
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def unpatchify(x, type='out'):
    """
    out is of shape [N, p*p*3, h, w]
    mask is of shape [N, h*w, p*p*3]
    imgs: (N, 3, H, W)
    """

    p = 32
    h = w = 128//p

    if type == 'out':
        n, c, _, _ = x.shape
        x = x.reshape(n, c, -1)
        x = torch.einsum('ncp->npc', x)

    h = w = int(x.shape[1]**.5)
    # h = w = self.img_size // p
    assert h * w == x.shape[1]
    
    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    return imgs

def run_one_image(img, model, id, type_s2):
    x = torch.from_numpy(img).unsqueeze(0).float().cuda()
    loss, y, mask = model({'sentinel2':x}, mask_ratio=0.6)

    y = y['sentinel2']
    y = unpatchify(y, 'out')
    y = torch.einsum('bchw->bhwc', y).detach().cpu()

    
    mask = mask.detach()
    p = model.patch_size
    mask = mask.unsqueeze(-1).repeat(1, 1, p**2 *3)  # (N, H*W, p*p*3)
    mask = unpatchify(mask, 'mask')  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    
    x = torch.einsum('nchw->nhwc', x)

    # masked image
    x = x.detach().cpu()
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 24]

    plt.subplot(1, 4, 1)
    show_image(x[0], "original", type_s2)

    plt.subplot(1, 4, 2)
    show_image(im_masked[0], "masked", type_s2)

    plt.subplot(1, 4, 3)
    show_image(y[0], "reconstruction", type_s2)

    plt.subplot(1, 4, 4)
    show_image(im_paste[0], "reconstruction + visible", type_s2)

    plt.savefig(f'masking_visualizations/masking_process_{id}_1M.png', bbox_inches='tight', pad_inches=0.1)


def prepare_model(checkpoint_dir, args):

    args.modalities = MODALITIES
    args.modalities_full = MODALITIES_FULL
    args.batch_size = 1
    args.data_path = '/home/qbk152/vishal/global-lr/data/data_1M_130_new/data_1M_130_new.h5'
    args.data_name = args.data_path.split('.')[0].split('/')[-1]
    args.splits_path = args.data_path.split('.')[0] + '_splits.json'
    args.tile_info_path = args.data_path.split('.')[0] + '_tile_info.json'
    args.band_stats_path = args.data_path.split('.')[0] + '_band_stats.json'
    # quick check to see if all the files exist
    assert os.path.exists(args.data_path), "Data file does not exist"
    assert os.path.exists(args.splits_path), "Split file does not exist"
    assert os.path.exists(args.tile_info_path), "Tile info file does not exist"
    assert os.path.exists(args.band_stats_path), "Band stats file does not exist"

    args.band_stats = json.load(open(args.band_stats_path, 'r'))
    args.tile_info = json.load(open(args.tile_info_path, 'r'))

    
    checkpoint = torch.load(checkpoint_dir, map_location='cpu')
    model = fcmae.__dict__['convnextv2_pico'](
        mask_ratio=0.6,
        decoder_depth=1,
        decoder_embed_dim=512,
        norm_pix_loss=True,
        args = args
    )



    msg = model.load_state_dict(checkpoint['model'], strict=False)

    print(msg)

    return model.cuda()


if __name__ == '__main__':
    import json
    import h5py


    idxs = [445583, 1110227, 1041332, 20296]

    args = get_args_parser().parse_args()
    tile_info = json.load(open('/home/qbk152/vishal/global-lr/data/data_1M_130_new/data_1M_130_new_tile_info.json'))
    file = h5py.File('/home/qbk152/vishal/global-lr/data/data_1M_130_new/data_1M_130_new.h5', 'r')
    # checkpoint_dir = '/home/qbk152/vishal/global-lr-train/ConvNeXt-V2/results/pre-training-s2-full-loss/checkpoint-429.pth'
    # checkpoint_dir = '/home/qbk152/vishal/global-lr-train/ConvNeXt-V2/results/pre-training-s2-full-loss-l2/checkpoint-1599.pth'
    # checkpoint_dir = '/home/qbk152/vishal/global-lr-train/ConvNeXt-V2/results/pre-training-s2-full-loss/checkpoint-1599.pth'
    checkpoint_dir = '/home/qbk152/vishal/global-lr-train/ConvNeXt-V2/results/pre-training-s2-1M-L2-8gpu/checkpoint-1306.pth'


    s2 = file['sentinel2']
    meta = file['metadata']
    model = prepare_model(checkpoint_dir, args)

    for idx in idxs:

        img = np.array(s2[idx][[1, 2, 3], :, :])
        name = meta[idx][0].decode('utf-8')
        print('processing image', name)
        type_s2 = tile_info[name]['S2_type']
        print('type of S2 image', type_s2)
        if tile_info[name]['S2_type'] == 'l2a':
            mean = np.array(mean_l2a)
            std = np.array(std_l2a)
        elif tile_info[name]['S2_type'] == 'l1c':
            mean = np.array(mean_l1c)
            std = np.array(std_l1c)

        img = (img - mean[:, None, None]) / std[:, None, None]

        # torch seed everything
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        run_one_image(img, model, idx, type_s2)




