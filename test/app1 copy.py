from baseline_BBDM.BBDM_folk.model.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel as LBBDM
from flask import Flask, render_template, request, flash, redirect
import os
import matplotlib.pyplot as plt

import io
import base64

from ema import EMA

import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
import pydicom
import torchvision.transforms as transforms
import yaml
import argparse
import omegaconf 
import torch

app = Flask(__name__)
app.secret_key = "super secret key"

org_out = './upload/'
device_1 = 'cuda:0'

use_ema = True

f = open('/home/vaipe/PET_CT/CT2PET_UI/baseline_BBDM/BBDM_folk/configs/LBBDMxVq13.yaml', 'r')
dict_config = yaml.load(f, Loader=yaml.FullLoader)

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict) or isinstance(value, omegaconf.dictconfig.DictConfig):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


# load_model 
weight_path = "/home/vaipe/PET_CT/CT2PET_UI/baseline_BBDM/baseline.pth"
model_states = torch.load(weight_path, map_location='cpu')

nconfig = dict2namespace(dict_config)
# baseline_BBDM = LBBDM(nconfig.model).to(device_1)
baseline_BBDM = LBBDM(nconfig.model)
baseline_BBDM = baseline_BBDM.to(device_1)
if use_ema: 
    ema = EMA(nconfig.model.EMA.ema_decay)
    ema.register(baseline_BBDM)

baseline_BBDM.load_state_dict(model_states['model'])

if use_ema:
    ema.shadow = model_states['ema']
    ema.reset_device(baseline_BBDM)

baseline_BBDM = baseline_BBDM.to(device_1)
if use_ema:
    ema.apply_shadow(baseline_BBDM)


baseline_BBDM.eval()

@app.route('/', methods=['GET', 'POST'])
def get_info():
    if request.method == "GET":
        return render_template('form_image1.html', file='')
    if request.method == "POST":
        uploaded_file = request.files["formFile"]
        dicom_data = pydicom.dcmread(uploaded_file, force=True)
        pixel_data = dicom_data.pixel_array
        pixel_data = pixel_data / 2047.
        
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
     
        image = Image.fromarray(pixel_data) 
        image = transform(image)
        x_cond = image.unsqueeze(0).to(device_1)
        
        with torch.no_grad():
            # check device
            print('device:', x_cond.device)

            sample = baseline_BBDM.sample(x_cond, clip_denoised=False)
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
    port = int(os.environ.get("PORT", 9876))
    app.run(debug=False, host="0.0.0.0", port=port)


# {% comment %} {% if img_data %}
#             <img id="outputImage" src="data:image/png;base64,{{ img_data }}" class="img-fluid" />
#     {% endif %} {% endcomment %}