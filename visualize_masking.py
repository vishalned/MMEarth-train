# a file to help visualizing the masking and output of the model.

import torch
import numpy as np

import matplotlib.pyplot as plt
import models.fcmae as fcmae
from main_pretrain import get_args_parser
from MODALITIES import *
import os
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import ListedColormap
import json
import sys



stats_path = '/home/qbk152/vishal/global-lr/data/data_1M_130_new/data_1M_130_new_band_stats.json'
stats_full = json.load(open(stats_path, 'r'))
mean_l2a = np.array(stats_full['sentinel2_l2a']['mean'])[[1, 2, 3]]
std_l2a = np.array(stats_full['sentinel2_l2a']['std'])[[1, 2, 3]]
mean_l1c = np.array(stats_full['sentinel2_l1c']['mean'])[[1, 2, 3]]
std_l1c = np.array(stats_full['sentinel2_l1c']['std'])[[1, 2, 3]]

mean_aster = np.array(stats_full['aster']['mean'])[0]
std_aster = np.array(stats_full['aster']['std'])[0]

mean_canopy = np.array(stats_full['canopy_height_eth']['mean'])[0]
std_canopy = np.array(stats_full['canopy_height_eth']['std'])[0]


s1_band = [1]
mean_s1 = np.array(stats_full['sentinel1']['mean'])[s1_band]
std_s1 = np.array(stats_full['sentinel1']['std'])[s1_band]


min_aster = stats_full['aster']['min'][0]
max_aster = stats_full['aster']['max'][0]

min_canopy = stats_full['canopy_height_eth']['min'][0]
max_canopy = stats_full['canopy_height_eth']['max'][0]



min_s1 = stats_full['sentinel1']['min'][1]
max_s1 = stats_full['sentinel1']['max'][1]


channels = {
    'sentinel2':12,
    'aster':2,
    'canopy_height_eth':2,
    'sentinel1':8,
    'dynamic_world':1,
    'esa_worldcover':1
}

def show_image(img, title='', type='', mask = None, x = None):
    # assert img.shape[2] == 3
    if type == 'dynamic_world' or type == 'esa_worldcover':
        if type == 'dynamic_world':
            
            colors = ['#419bdf', '#397d49', '#88b053', '#7a87c6', '#e49635', '#dfc35a', '#c4281b', '#a59b8f', '#b39fe1']
            norm = BoundaryNorm([0, 1, 2, 3, 4, 5, 6, 7, 8], len(colors))

            cmap = ListedColormap(colors)

            img = img.reshape(112, 112)
            if mask is not None:
                mask = mask.squeeze()
                img[mask == 1] = np.nan
                cmap.set_bad(color='black')

            plt.imshow(img, cmap=cmap, norm=norm)
            plt.title(title, fontsize=16)

            plt.axis('off')
            return
        else:
            colormap = [
            '#006400',  # Tree cover - 10
            '#ffbb22',  # Shrubland - 20
            '#ffff4c',  # Grassland - 30
            '#f096ff',  # Cropland - 40
            '#fa0000',  # Built-up - 50
            '#b4b4b4',  # Bare / sparse vegetation - 60
            '#f0f0f0',  # Snow and ice - 70
            '#0064c8',  # Permanent water bodies - 80
            '#0096a0',  # Herbaceous wetland - 90
            '#00cf75',  # Mangroves - 95
            '#fae6a0',   # Moss and lichen - 100
            # '#000000'   # No data - 255
            ]

            # bounds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
            bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            norm = BoundaryNorm(bounds, len(colormap))



            cmap = ListedColormap(colormap)
            if mask is not None:
                mask = mask.squeeze(0)
                img[mask == 1] = np.nan
                cmap.set_bad(color='black')
            img = img.reshape(112, 112)
            plt.imshow(img, cmap=cmap, norm=norm)
            plt.title(title, fontsize=16)
            plt.axis('off')
            return

    if type == 'sentinel2_l2a':
        mean = mean_l2a
        std = std_l2a
    elif type == 'sentinel2_l1c':
        mean = mean_l1c
        std = std_l1c
    elif type == 'aster':
        mean = mean_aster
        std = std_aster
    elif type == 'canopy_height_eth':
        mean = mean_canopy
        std = std_canopy
    elif type == 'sentinel1':
        mean = mean_s1
        std = std_s1
    else:
        mean = None
        std = None
    x = x[0]
    img = img.permute(2, 0, 1)
    x = x.permute(2, 0, 1)

    
    if mean is not None and std is not None:
        # check if mean and std are single values or arrays
        if isinstance(mean, (int, float)):

            img = img*std + mean
            x = x*std + mean
        else:
            img = img*std[:, None, None] + mean[:, None, None]
            x = x*std[:, None, None] + mean[:, None, None]
    if 'sentinel2' in type:
        vmin, vmax = None, None
        img = img/10000 
        clip_val = 0.3
        img = np.clip(img, 0, clip_val)
        img = img / clip_val # clamp to 0-1
    elif 'sentinel1' in type:
        img = (np.clip(img, -30, 0) + 30) / 30
        x = (np.clip(x, -30, 0) + 30) / 30
        vmin, vmax = None, None
        # vmin, vmax = torch.min(x).item(), torch.max(x).item()

    if 'aster' in type:
        # x = x - min_aster
        # x = x / (max_aster - min_aster)
        # vmin, vmax = torch.min(x).item(), torch.max(x).item()
        img = img - min_aster
        img = img / (max_aster - min_aster)
        # img = img.clip(0, 1)
        # vmin, vmax = torch.min(img).item(), torch.max(img).item()
        vmin, vmax = None, None
    elif 'canopy' in type:
        x = x - min_canopy
        x = x / (max_canopy - min_canopy)
        vmin, vmax = torch.min(x).item(), torch.max(x).item()
        img = img - min_canopy
        img = img / (max_canopy - min_canopy)

    # apply the mask again 
    


    img = torch.einsum('chw->hwc', img)
    if mask is not None:
        print(img.shape, mask.shape)
        mask = mask.squeeze(0)
        img[mask == 1] = np.nan
        cm = plt.cm.get_cmap('viridis')
        cm.set_bad(color='black')
    else:
        cm = None

    # sentinel2 image is in the order of b2, b3, b4. We need to change it to b4, b3, b2 since that is the order of RGB
    img = img[:, :, [2, 1, 0]] if 'sentinel2' in type else img
    # if mask is not None, we want to make the masked part of the image to be black
    # if mask is not None:
    #     # print(mask.shape)
    #     # mask = torch.einsum('nchw->nhwc', mask)
    #     # print(mask.shape)
    #     # mask = mask[:, :, :, [2, 1, 0]] if 'sentinel2' in type else mask
    #     mask = mask.numpy().squeeze()
    #     img = img.numpy()
    #     print(img.shape, mask.shape)
    #     # img = img.squeeze()
    #     if 'dynamic_world' in type or 'esa_worldcover' in type:
    #         img[mask == 0] = 9 if 'dynamic_world' in type else 11
    #     else:
    #         img[mask == 1] = 0 
        # img = torch.from_numpy(img)


    plt.imshow(img, cmap=cm, vmin=vmin, vmax=vmax)
    plt.title(title, fontsize=16) if title != '' else None
    plt.xticks([])
    plt.yticks([])
    return

def unpatchify(x, type=''):
    """
    out is of shape [N, p*p*3, h, w]
    mask is of shape [N, h*w, p*p*3]
    imgs: (N, 3, H, W)
    """

    p = 16
    h = w = 112//p

    if 'mask' not in type:
        n, c, _, _ = x.shape
        x = x.reshape(n, c, -1)
        x = torch.einsum('ncp->npc', x)
    else:
        type = type.split('-')[1]

    h = w = int(x.shape[1]**.5)
    # h = w = self.img_size // p
    assert h * w == x.shape[1]
    
    x = x.reshape(shape=(x.shape[0], h, w, p, p, channels[type]))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], channels[type], h * p, h * p))
    return imgs

def run_one_image(img, data_dict, model, id, type_s2):
    # x = torch.from_numpy(img).unsqueeze(0).float().cuda()
    # convert all the data to cuda
    for key in data_dict.keys():
        data_dict[key] = data_dict[key].cuda()
    loss, y, mask_orig, _, _, _ = model(data_dict, mask_ratio=0.6)
    
    pred_keys = ['sentinel2', 'canopy_height_eth', 'sentinel1', 'dynamic_world', 'esa_worldcover']

    plt.rcParams['figure.figsize'] = [28, 16]
    # make a plot with the top row showing original and bottom row showing the reconstruction
    for idx, key in enumerate(pred_keys):
        if key in y:
            print('running for', key)

            y[key] = y[key].detach().cpu()
            if 'dynamic_world' in key or 'esa_worldcover' in key:
                # these are classification tasks. We need to take the argmax of the output after applying softmax
                from torch.nn.functional import softmax
                n, c, _, _ = y[key].shape
                y[key] = y[key].reshape(1, c, -1)
                y[key] = torch.einsum('ncp->npc', y[key])
                y[key] = y[key].reshape(1, 7, 7, 16, 16, -1)
                y[key] = torch.einsum('nhwpqc->nchpwq', y[key])
                c_ = 9 if 'dynamic_world' in key else 11
                y[key] = y[key].reshape(1, c_, 112, 112)
                y[key] = torch.einsum('nchw->nhwc', y[key])
                y[key] = softmax(y[key], dim=-1)
                y[key] = torch.argmax(y[key], dim=-1)
                y[key] = y[key].reshape(shape=(1, 1, 112, 112))

            else:
                y[key] = unpatchify(y[key], key)
            y[key] = torch.einsum('nchw->nhwc', y[key]).detach().cpu()

            mask = mask_orig.detach()
            p = model.patch_size
            mask = mask.unsqueeze(-1).repeat(1, 1, p**2 * channels[key])
            mask = unpatchify(mask, f'mask-{key}')  # 1 is removing, 0 is keeping
            mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

            
            x = data_dict[key].detach().cpu()
            print('original shape', x.shape)
            x = torch.einsum('nchw->nhwc', x)
            
            im_masked = x * (1 - mask)
            im_paste = x * (1 - mask) + y[key] * mask
            orig_masked = x * mask
        

            if key == 'sentinel2':
                
                x = x[:, :, :, [1, 2, 3]]
                im_masked = im_masked[:, :, :, [1, 2, 3]]
                y[key] = y[key][:, :, :, [1, 2, 3]]
                im_paste = im_paste[:, :, :, [1, 2, 3]]
                orig_masked = orig_masked[:, :, :, [1, 2, 3]]
                mask = mask[:, :, :, [1, 2, 3]]
                key = f'sentinel2_{type_s2}'

            elif key == 'aster':
                x = x[:, :, :, [0]]
                im_masked = im_masked[:, :, :, [0]]
                y[key] = y[key][:, :, :, [0]]
                im_paste = im_paste[:, :, :, [0]]
                orig_masked = orig_masked[:, :, :, [0]]
                mask = mask[:, :, :, [0]]

            elif key == 'canopy_height_eth':
                x = x[:, :, :, [0]]
                im_masked = im_masked[:, :, :, [0]]
                y[key] = y[key][:, :, :, [0]]
                im_paste = im_paste[:, :, :, [0]]
                orig_masked = orig_masked[:, :, :, [0]]
                mask = mask[:, :, :, [0]]
            elif key == 'sentinel1':
                # print the min and max of each band
                # for i in range(8):
                #     print(f'min and max of band {i} is {x[0, :, :, i].min(), x[0, :, :, i].max()}')
                s1_band = [1]
                x = x[:, :, :, s1_band]
                im_masked = im_masked[:, :, :, s1_band]
                y[key] = y[key][:, :, :, s1_band]
                im_paste = im_paste[:, :, :, s1_band]
                orig_masked = orig_masked[:, :, :, s1_band]
                mask = mask[:, :, :, s1_band]

            elif key == 'dynamic_world':
                x = x[:, :, :, :]
                im_masked = im_masked[:, :, :, :]
                y[key] = y[key][:, :, :, :]
                im_paste = im_paste[:, :, :, :]
                orig_masked = orig_masked[:, :, :, :]
                mask = mask[:, :, :, :]

            elif key == 'esa_worldcover':
                x = x[:, :, :, :]
                im_masked = im_masked[:, :, :, :]
                y[key] = y[key][:, :, :, :]
                im_paste = im_paste[:, :, :, :]
                orig_masked = orig_masked[:, :, :, :]
                mask = mask[:, :, :, :]

            plt_idx = idx + 5
            plt_idx2 = idx + 10
            plt_idx3 = idx + 15
            plt.subplot(4, 5, idx + 1)
            names = {
                'sentinel2_l2a': 'Sentinel-2 L2A',
                'sentinel2_l1c': 'Sentinel-2 L1C',
                'aster': 'ASTER Elevation',
                'canopy_height_eth': 'Canopy Height',
                'sentinel1': 'Sentinel-1 VH (Asc)',
                'dynamic_world': 'Dynamic World',
                'esa_worldcover': 'ESA World Cover'
            }
            show_image(im_masked[0], f"{names[key]}", key, mask, x)
            if idx == 0:
                plt.ylabel('Masked', fontsize=16)

            # add a ylabel to the first plot

            plt.subplot(4, 5, plt_idx + 1)
            if 'sentinel2' in key:
                show_image(y['sentinel2'][0], '', key, None, x)
            else:
                show_image(y[key][0], '', key, None, x)
            if idx == 0:
                plt.ylabel('Reconstruction', fontsize=16)


            plt.subplot(4, 5, plt_idx2 + 1)
            show_image(im_paste[0], '', key, None, x)
            if idx == 0:
                plt.ylabel('Reconstruction + Visible', fontsize=16)


            plt.subplot(4, 5, plt_idx3 + 1)
            show_image(x[0], '', key, None, x)
            if idx == 0:
                plt.ylabel('Original', fontsize=16)



           


    plt.savefig(f'masking_visualizations/masking_process_{id}.png', bbox_inches='tight', dpi=300, format='png')
    plt.savefig(f'masking_visualizations/masking_process_{id}.pdf', bbox_inches='tight', dpi=300, format='pdf')

    # sentinel2
    # y = y['sentinel2']
    # y = unpatchify(y, 'sentinel2')
    # y = torch.einsum('bchw->bhwc', y).detach().cpu()

    
    # mask = mask.detach()
    # p = model.patch_size
    # mask = mask.unsqueeze(-1).repeat(1, 1, p**2 * channels['sentinel2'])
    # mask = unpatchify(mask, 'mask')  # 1 is removing, 0 is keeping
    # mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    
    # x = torch.einsum('nchw->nhwc', x)

    # # masked image
    # x = x.detach().cpu()
    # im_masked = x * (1 - mask)

    # # MAE reconstruction pasted with visible patches
    # im_paste = x * (1 - mask) + y * mask

    # # for viewing, we only choose 3 bands 
    # x = x[:, :, :, [2, 3, 4]]
    # im_masked = im_masked[:, :, :, [2, 3, 4]]
    # y = y[:, :, :, [2, 3, 4]]
    # im_paste = im_paste[:, :, :, [2, 3, 4]]

    # # make the plt figure larger
    # plt.rcParams['figure.figsize'] = [24, 24]

    # plt.subplot(1, 4, 1)
    # show_image(x[0], "original", type_s2)

    # plt.subplot(1, 4, 2)
    # show_image(im_masked[0], "masked", type_s2)

    # plt.subplot(1, 4, 3)
    # show_image(y[0], "reconstruction", type_s2)

    # plt.subplot(1, 4, 4)
    # show_image(im_paste[0], "reconstruction + visible", type_s2)

    # plt.savefig(f'masking_visualizations/masking_process_{id}_1M.png', bbox_inches='tight', pad_inches=0.1)


def prepare_model(checkpoint_dir, args):

    args.inp_modalities = INP_MODALITIES
    args.out_modalities = OUT_MODALITIES
    args.random_crop = True
    args.modalities = args.inp_modalities.copy()
    args.modalities.update(args.out_modalities)

    args.modalities_full = MODALITIES_FULL
    args.batch_size = 1
    args.data_path = '/projects/dereeco/data/global-lr/data_1M_130_new/data_1M_130_new.h5'
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
    model = fcmae.__dict__['convnextv2_atto'](
        mask_ratio=0.6,
        decoder_depth=1,
        decoder_embed_dim=512,
        norm_pix_loss=True,
        patch_size=16,
        img_size=112,
        args = args
    )



    msg = model.load_state_dict(checkpoint['model'], strict=False)

    print(msg)

    return model.cuda()


if __name__ == '__main__':
    import json
    import h5py
    from custom_dataset import MultiModalDataset

    idxs = [445583, 1110227, 1041332, 20296]
    s2_type = ['l2a', 'l2a', 'l1c', 'l2a']
    # s1_bands = [[0, 1, 4, 5], [0, 1, 4, 5], [0, 1, 4, 5], 
    img_id = 3

    args = get_args_parser().parse_args()
    # tile_info = json.load(open('/projects/dereeco/data/global-lr/data_1M_130_new/data_1M_130_new_tile_info.json'))
    checkpoint_dir = '/projects/dereeco/data/global-lr/ConvNeXt-V2/results/pt-all_mod_uncertainty/checkpoint-199.pth'

    print('Preparing model')
    model = prepare_model(checkpoint_dir, args)

    print('Creating dataset loader')
    dataset = MultiModalDataset(args, split="train")

    print('Loading data')
    data_dict = dataset.__getitem__(idxs[img_id])


        # torch seed everything
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    img = data_dict['sentinel2'].numpy()

    # remove key id from data_dict
    ids = data_dict.pop('id')
    # add an extra 1 in front of the tensor to make it a batch of 1
    for key in data_dict.keys():
        data_dict[key] = data_dict[key].unsqueeze(0)

    print('Running one image')
    run_one_image(img, data_dict, model, idxs[img_id], s2_type[img_id])





