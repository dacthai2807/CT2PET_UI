from baseline_BBDM.BBDM_folk.model.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel as LBBDM

from ema import EMA

import warnings
warnings.filterwarnings("ignore")

import yaml
from utils import dict2namespace, create_to_gen_html
import torch

device_1 = 'cuda:0'

use_ema = True

f = open('/home/vaipe/PET_CT/CT2PET_UI/baseline_BBDM/BBDM_folk/configs/LBBDMxVq13.yaml', 'r')
dict_config = yaml.load(f, Loader=yaml.FullLoader)

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

def infer(x_cond_param):

    x_cond = x_cond_param.clone()
    x_cond = x_cond.unsqueeze(0).to(device_1)
    
    with torch.no_grad():
        # check device
        print('device:', x_cond.device)

        sample = baseline_BBDM.sample(x_cond, clip_denoised=False)
        sample = sample.squeeze(0)
        sample = sample.detach().clone()
        sample = sample.mul_(0.5).add_(0.5).clamp_(0, 1.)
        sample = sample.mul_(32767).add_(0.2).clamp_(0, 32767).permute(1, 2, 0).to('cpu').numpy()

        return sample

    # Return the template with the image string
    # return render_template('form_image1.html', file='Image Upload Succeed', img_data=img_str)

# {% comment %} {% if img_data %}
#             <img id="outputImage" src="data:image/png;base64,{{ img_data }}" class="img-fluid" />
#     {% endif %} {% endcomment %}