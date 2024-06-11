import matplotlib.pyplot as plt

import io
import base64

import torch
import yaml
import argparse
import omegaconf
import numpy as np
import os 
from ema import EMA

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict) or isinstance(value, omegaconf.dictconfig.DictConfig):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

from segmented_conditional_BBDM.BBDM_folk.model.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel as LBBDM
import torchvision.transforms as transforms
IMAGE_SIZE = 256
from PIL import Image

SEGMENTATION_CHECKPOINT_FILE_PATH = "/mnt/disk1/PET_CT/CT2PET_UI/segmented_conditional_BBDM/Unet_resnet34/lightning_logs/version_0/checkpoints/epoch=3-step=6000.ckpt"


org_out = './upload/'
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



@app.route('/', methods=['GET', 'POST'])
def get_info():
    if request.method == "GET":
        return render_template('form_image1.html', file='')
    if request.method == "POST":
        uploaded_file = request.files["formFile"]
        dicom_data = pydicom.dcmread(uploaded_file, force=True)
        pixel_data = dicom_data.pixel_array

# np_ct_image = np.load(ct_path, allow_pickle=True) # get numpy array 
        with torch.no_grad(): 
            np_ct_image = pixel_data / float(2047) # normalize
            ct_image = Image.fromarray(np_ct_image) 

            x_cond = transform(ct_image) 
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

        # create a figure and plot sample on it 
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(sample, cmap='gray')
        ax.axis('off')
        
        # Convert the Matplotlib figure to a PNG image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        img_png = buf.getvalue()
        
        # Encode the PNG image to base64 string
        img_str = base64.b64encode(img_png).decode('utf-8')
        
        buf.close()
        plt.close(fig)  # Close the figure to free memory

        # Return the template with the image string
        return render_template('form_image1.html', file='Image Upload Succeed', img_data=img_str)



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 9999))
    app.run(debug=False, host="0.0.0.0", port=port)