'''
A script to define the model architecture and load the pre-trained weights. 
This script is used by the `torch.hub.load` function to load the model easily.
We only make the finetuning of the model possiible here. 

model files by default are stored in '~/.cache/torch/hub/checkpoints/checkpoint-199.pth'

kwargs:
    - num_classes: int, number of classes in the dataset
'''


import torch
import models.convnextv2 as convnextv2


dependencies = ['torch']

def MPMAE(model_size = 'convnextv2_atto', pretrained=True, device= 'cuda', **kwargs):
    """
    MPMAE model architecture
    """
    model = convnextv2.__dict__[model_size](**kwargs)
    ckpt_urls = {
        'pt-all_mod_atto_1M_64_uncertainty_56-8': 'https://sid.erda.dk/share_redirect/g23YOnaaTp/pt-all_mod_atto_1M_64_uncertainty_56-8/checkpoint-199.pth',
        'pt-all_mod_atto_1M_64_unweighted_56-8': 'https://sid.erda.dk/share_redirect/g23YOnaaTp/pt-all_mod_atto_1M_64_unweighted_56-8/checkpoint-199.pth',
        'pt-all_mod_atto_1M_128_uncertainty_112-16': 'https://sid.erda.dk/share_redirect/g23YOnaaTp/pt-all_mod_atto_1M_128_uncertainty_112-16/checkpoint-199.pth',
        'pt-S2_atto_1M_64_uncertainty_56-8': 'https://sid.erda.dk/share_redirect/g23YOnaaTp/pt-S2_atto_1M_64_uncertainty_56-8/checkpoint-199.pth'
    }

    if pretrained:
        if device == 'cpu':
            state_dict = torch.hub.load_state_dict_from_url(ckpt_urls['pt-all_mod_atto_1M_64_uncertainty_56-8'], map_location='cpu')
            model.load_state_dict(state_dict, map_location=torch.device('cpu'))
        else:
            if torch.cuda.is_available():
                state_dict = torch.hub.load_state_dict_from_url(ckpt_urls['pt-all_mod_atto_1M_64_uncertainty_56-8'])
                model.load_state_dict(state_dict)
            else:
                raise AssertionError("CUDA is not available. Set device='cpu' to load the model on CPU.")
    else:
        raise AssertionError("Loading the model with torch hub without pretrained weights is not supported.")
    return model
