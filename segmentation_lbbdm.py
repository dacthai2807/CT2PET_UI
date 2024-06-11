import matplotlib.pyplot as plt
import torch
import yaml

import numpy as np
from ema import EMA
from utils import dict2namespace, create_to_gen_html

from segmented_conditional_BBDM.BBDM_folk.model.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel as LBBDM
import torchvision.transforms as transforms
IMAGE_SIZE = 256

SEGMENTATION_CHECKPOINT_FILE_PATH = "/mnt/disk1/PET_CT/CT2PET_UI/segmented_conditional_BBDM/Unet_resnet34/lightning_logs/version_0/checkpoints/epoch=3-step=6000.ckpt"


device_1 = 'cuda:0'

use_ema = True

f = open('/mnt/disk1/PET_CT/CT2PET_UI/segmented_conditional_BBDM/conditional_LBBDM.yaml', 'r')

# load_model 
weight_path = '/mnt/disk1/PET_CT/CT2PET_UI/segmented_conditional_BBDM/segmented_guide_lbbdm_52.pth'
model_states = torch.load(weight_path, map_location='cpu')

dict_config = yaml.load(f, Loader=yaml.FullLoader)
nconfig = dict2namespace(dict_config)

ltbbdm = LBBDM(nconfig.model)
ltbbdm = ltbbdm.to(device_1)
if use_ema: 
    ema = EMA(nconfig.model.EMA.ema_decay)
    ema.register(ltbbdm)
ltbbdm.load_state_dict(model_states['model'])

if use_ema:
    ema.shadow = model_states['ema']
    ema.reset_device(ltbbdm)

ltbbdm = ltbbdm.to(device_1)
if use_ema:
    ema.apply_shadow(ltbbdm)

ltbbdm.eval()

# preprocess CT image  

transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.0),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor()
        ])
# segemntation model 
from segmented_conditional_BBDM.BBDM_folk.test_segmentor import SegmentationModel
model_name = "Unet"
encoder_name = "resnet34"
model = SegmentationModel.load_from_checkpoint(SEGMENTATION_CHECKPOINT_FILE_PATH, arch=model_name, encoder_name=encoder_name, in_channels=3, out_classes=1)
model = model.to(device_1)
model.eval()

from flask import Flask, render_template, request, flash, redirect
import pydicom
app = Flask(__name__)
app.secret_key = "super secret key"
###




def infer(x_cond_param):
# np_ct_image = np.load(ct_path, allow_pickle=True) # get numpy array 
    with torch.no_grad():
        
        x_cond = x_cond_param.clone()
        x_cond = x_cond.unsqueeze(0).to(device_1)

        x_cond_latent = ltbbdm.encode(x_cond)

        # extract segmentation mask
        ct_latent = (x_cond_latent / 4. + 0.5).clamp(0., 1.)

        np_ct_latent = ct_latent.squeeze().permute(1, 2, 0).cpu().numpy()
        np_ct_latent = (np_ct_latent * 255.).astype(np.uint8)

        ct_latent = torch.from_numpy(np_ct_latent).permute(2, 0, 1).to(device_1)
        
        logits = model(ct_latent) 
        add_cond = logits.sigmoid()

        ###
        print('device x_cond:', x_cond_latent.device) 
        print('device: add_cond', add_cond.device)

        # ltbbdm = ltbbdm.to(device_1)

        temp = ltbbdm.p_sample_loop(y=x_cond_latent,
                                    add_cond=add_cond,
                                    context=None,
                                    clip_denoised=False,
                                    sample_mid_step=False)
        
        x_latent = temp
        sample = ltbbdm.decode(x_latent)
        sample = sample.squeeze(0)
        sample = sample.detach().clone()
        sample = sample.mul_(0.5).add_(0.5).clamp_(0, 1.)
        sample = sample.mul_(32767).add_(0.2).clamp_(0, 32767).permute(1, 2, 0).to('cpu').numpy()

        return sample
    # Return the template with the image string
#    /\ return render_template('form_image1.html', file='Image Upload Succeed', img_data=img_str)

