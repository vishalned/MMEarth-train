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
from helpers import remap_checkpoint_keys, load_state_dict
from timm.models.layers import trunc_normal_


dependencies = ['torch', 'timm']

def load_custom_checkpoint(model, checkpoint, linear_probe):
    if not linear_probe: # finetuning
        if "unet" in model:
            raise ValueError(
                "All experiments were done with a combination of linear probe and fine-tuning. Please set the --linear_probe to True, to enable linear probe."
            )
        checkpoint_model = checkpoint["model"] if "model" in checkpoint else checkpoint
        state_dict = model.state_dict()
        for k in ["head.weight", "head.bias"]:
            if (
                k in checkpoint_model
                and checkpoint_model[k].shape != state_dict[k].shape
            ):
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # remove decoder weights
        checkpoint_model_keys = list(checkpoint_model.keys())
        for k in checkpoint_model_keys:
            if "decoder" in k or "mask_token" in k or "proj" in k or "pred" in k:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]


        checkpoint_model = remap_checkpoint_keys(checkpoint_model)
        load_state_dict(model, checkpoint_model)
        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)
        torch.nn.init.constant_(model.head.bias, 0.)


    elif linear_probe: # linear probe
        # we still start with the same fine-tuning pre-trained model, and then remove the head. we then make the model frozen, and add a new head for linear probe

        checkpoint_model = checkpoint["model"] if "model" in checkpoint else checkpoint
        state_dict = model.state_dict()
        for k in ["head.weight", "head.bias"]:
            if (
                k in checkpoint_model
                and checkpoint_model[k].shape != state_dict[k].shape
            ):
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # remove decoder weights
        checkpoint_model_keys = list(checkpoint_model.keys())
        for k in checkpoint_model_keys:
            if "decoder" in k or "mask_token" in k or "proj" in k or "pred" in k:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        checkpoint_model = remap_checkpoint_keys(checkpoint_model)
        load_state_dict(model, checkpoint_model)

    return model

def MPMAE(model = 'convnextv2_atto', ckpt_name = 'pt-all_mod_atto_1M_64_uncertainty_56-8', pretrained=True, linear_probe=True, in_chans = 12, **kwargs):
    """
    MPMAE model architecture
    """
    model = convnextv2.__dict__[model](**kwargs)
    ckpt_urls = {
        'pt-all_mod_atto_1M_64_uncertainty_56-8': 'https://sid.erda.dk/share_redirect/g23YOnaaTp/pt-all_mod_atto_1M_64_uncertainty_56-8/checkpoint-199.pth',
        'pt-all_mod_atto_1M_64_unweighted_56-8': 'https://sid.erda.dk/share_redirect/g23YOnaaTp/pt-all_mod_atto_1M_64_unweighted_56-8/checkpoint-199.pth',
        'pt-all_mod_atto_1M_128_uncertainty_112-16': 'https://sid.erda.dk/share_redirect/g23YOnaaTp/pt-all_mod_atto_1M_128_uncertainty_112-16/checkpoint-199.pth',
        'pt-S2_atto_1M_64_uncertainty_56-8': 'https://sid.erda.dk/share_redirect/g23YOnaaTp/pt-S2_atto_1M_64_uncertainty_56-8/checkpoint-199.pth'
    }

    if pretrained:
        # download the pretrained weights and get the path
        checkpoint = torch.hub.load_state_dict_from_url(ckpt_urls[ckpt_name], map_location='cpu')
        model = load_custom_checkpoint(model, checkpoint, linear_probe)
    else:
        raise AssertionError("Loading the model with torch hub without pretrained weights is not supported.")
    return model
