from baseline_BBDM.BBDM_folk.model.BrownianBridge.LatentBrownianBridgeModel import LatentBrownianBridgeModel as LBBDM
from flask import Flask, render_template, request, flash, redirect
import os
# from model import *
import numpy as np
from PIL import Image
from datetime import datetime
import cv2
import warnings
warnings.filterwarnings("ignore")
import pydicom
import torchvision.transforms as transforms
import yaml
import argparse
import omegaconf 

app = Flask(__name__)
app.secret_key = "super secret key"

org_out = './upload/'
device_1 = 'cuda:0'
device_2 = 'cuda:2'

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

nconfig = dict2namespace(dict_config)
baseline_BBDM = LBBDM(nconfig.model)
baseline_BBDM.eval()
# detection = DETECT(weight_path="./weights/craft_v1.pth", device=device_1)
# detection = DETECT(device=device_2)
# recognition = RECOGNIZE(device=device_2)
# temp_classify = TEMP_CLASSIFY(weight_path='./weights/template_classify_mbn.pth', device='cpu')

# gcn_chungnhan = INFO_EXTRACT(weight_path='./weights/layout_v2_final_gcn_chungnhan_2022-11-27.pth', template='gcn_chungnhan', device=device_2)
# gcn_nhao_dato = INFO_EXTRACT(weight_path='./weights/layout_v2_final_gcn_nhao_dato_2023-01-17.pth', template='gcn_nhao_dato', device=device_1)
# gcn_qsdd_mst = INFO_EXTRACT(weight_path='./weights/layout_v2_final_gcn_qsdd_mst_2023-06-23.pth', template='gcn_qsdd_mst', device=device_2)
# gcn_qsdd_mt = INFO_EXTRACT(weight_path='./weights/layout_v2_final_gcn_qsdd_mt_2023-01-20.pth', template='gcn_qsdd_mt', device=device_2)
# gcn_qsdd_msp = INFO_EXTRACT(weight_path ='./weights/layout_v2_final_gcn_qsdd_msp_2022-11-27.pth', template='gcn_qsdd_msp', device=device_1)
# gcn_qsdd_tsgl_mc = INFO_EXTRACT(weight_path ='./weights/layout_v2_final_gcn_qsdd_tsgl_mc_2022-11-27.pth', template='gcn_qsdd_tsgl_mc', device=device_2)
# gcn_qsdd_tsgl_ms = INFO_EXTRACT(weight_path ='./weights/layout_v2_final_gcn_qsdd_tsgl_ms_2023-05-21.pth', template='gcn_qsdd_tsgl_ms', device=device_2)
# gcn_qsdd_tsgl_mt = INFO_EXTRACT(weight_path ='./weights/layout_v2_final_gcn_qsdd_tsgl_mt_2023-02-07.pth', template='gcn_qsdd_tsgl_mt', device=device_2)

def get_model_extract(image):
    template = temp_classify.get_type_template(image)
    if template == 'gcn_chungnhan':
        return gcn_chungnhan, 'gcn_chungnhan'
    if template == 'gcn_nhao_dato':
        return gcn_nhao_dato, 'gcn_nhao_dato'
    if template == 'gcn_qsdd_msp':
        return gcn_qsdd_msp, 'gcn_qsdd_msp'
    if template == 'gcn_qsdd_mst':
        return gcn_qsdd_mst, 'gcn_qsdd_mst'
    if template == 'gcn_qsdd_mt':
        return gcn_qsdd_mt, 'gcn_qsdd_mt'
    if template == 'gcn_qsdd_tsgl_mc':
        return gcn_qsdd_tsgl_mc, 'gcn_qsdd_tsgl_mc'
    if template == 'gcn_qsdd_tsgl_ms_table_1':
        return gcn_qsdd_tsgl_ms_table_1, 'gcn_qsdd_tsgl_ms_table_1'
    if template == 'gcn_qsdd_tsgl_ms_table_3':
        return gcn_qsdd_tsgl_ms_table_3, 'gcn_qsdd_tsgl_ms_table_3'
    if template == 'gcn_qsdd_tsgl_ms':
        return gcn_qsdd_tsgl_ms, 'gcn_qsdd_tsgl_ms'
    if template == 'gcn_qsdd_tsgl_mt':
        return gcn_qsdd_tsgl_mt, 'gcn_qsdd_tsgl_mt'
    else:
        return 'Ko ho tro', 0

@app.route('/', methods=['GET', 'POST'])
def get_info():
    if request.method == "GET":
        return render_template('form_image.html', file='')
    if request.method == "POST":
        uploaded_file = request.files["formFile"]
        dicom_data = pydicom.dcmread(uploaded_file)
        pixel_data = dicom_data.pixel_array
        pixel_data = pixel_data / 2047.
        
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
     
        # image = Image.open(uploaded_file)
        # image = image.convert("RGB")
        # image = image.resize((int(1000*image.size[0]/image.size[1]), 1000))
        # img = np.array(image)
        image = Image.fromarray(pixel_data) 
        image = transform(image)
        x_cond = image.to(device_1)
        
        sample = baseline_BBDM.sample(x_cond, clip_denoised=False)
        
        print(sample.min(), sample.max(), sample.shape)
        
        return
        begin = datetime.now()
        print('===== start =====')
        # template classify
        start_time = datetime.now()
        try:
            info_extract, template_type = get_model_extract(image)
        except:
            filename = os.path.join(org_out, 'error', datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + ".jpg")
            img_save = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(filename, img_save)
            return render_template('form_image.html',
                        file='Image Upload Succeed',
                        notice="Template không hỗ trợ")
        if template_type == 0:
            filename = os.path.join(org_out, 'other', datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + ".jpg")
        else:  
            filename = os.path.join(org_out, str(template_type), datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + ".jpg")
        img_save = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(filename, img_save)
        print('template classify', datetime.now() - start_time)
        # Check if template is other
        if template_type == 0:
            return render_template('form_image.html',
                        file='Image Upload Succeed',
                        notice="Template không hỗ trợ")
        
        # bright and denoise
        start_time = datetime.now()
        brighten_denoise = BRIGHTEN_DENOISE(img)
        brighten_denoise_img = brighten_denoise.brighten_and_denoise_image()
        print('brighten_denoise image', datetime.now() - start_time)

        # Rotate
        start_time = datetime.now()
        list_phrase = detection.bbox_detect_phrase(brighten_denoise_img, read_image=False)
        angle_detect = ANGLE_DETECT(brighten_denoise_img, [list_phrase])
        final_angle, rotated_img = angle_detect.final_detect(0)
        width, height = rotated_img.shape[1], rotated_img.shape[0]
        print('rotate image', datetime.now() - start_time)
        
        # detect
        start_time = datetime.now()
        list_bbox = detection.bbox_detect(rotated_img)
        print('detect time', datetime.now() - start_time)
        
        # recog
        start_time = datetime.now()
        processed_annos = recognition.bbox_recognize(list_bbox, width, height)
        print('recog time', datetime.now() - start_time)
        
        # infor extract
        start_time = datetime.now()
        out_path = os.path.join(org_out, template_type, datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + ".json")
            
        img_for_PIL = (rotated_img * 255).astype(np.uint8)
        image = Image.fromarray(img_for_PIL)
        img_data, informations = info_extract.get_information(processed_annos, image, out_path, saved=False)
        if 'gcn_chungnhan' in template_type:
            info = extract_gcn_chungnhan(informations)
        if 'gcn_nhao_dato' in template_type:
            info = extract_gcn_nhao_dato(informations)
        if 'msp' in template_type:
            info = extract_gcn_qsdd_msp(informations)
        if 'mst' in template_type:
            info = extract_gcn_qsdd_mst(informations)
        if 'gcn_qsdd_mt' == template_type:
            info = extract_gcn_qsdd_mt(informations)
        if 'mc' in template_type:
            info = extract_gcn_qsdd_tsgl_mc(informations)
        if 'ms_table_1' in template_type:
            info = extract_gcn_qsdd_tsgl_ms_table_1(informations)
        if 'ms_table_3' in template_type:
            info = extract_gcn_qsdd_tsgl_ms_table_3(informations)
        if 'gcn_qsdd_tsgl_ms' == template_type:
            info = extract_gcn_qsdd_tsgl_ms(informations)
        if 'gcn_qsdd_tsgl_mt' == template_type:
            info = extract_gcn_qsdd_tsgl_mt(informations)
        
        print('kie and draw and post time', datetime.now() - start_time)
        total = datetime.now() - begin
        print('total time consuming', datetime.now() - begin)
        print('===== done =====')
        data = []
        # print(info)
        data.append(info)
        print(template_type)
        
        return render_template('form_image.html',
                        file='Image Upload Succeed',
                        img_data=img_data,
                        template_type=template_type,
                        informations = data,
                        time_consume=total)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 9876))
    app.run(debug=False, host="0.0.0.0", port=port)
