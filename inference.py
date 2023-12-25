import os
from model import *
import numpy as np
from PIL import Image
from datetime import datetime
import cv2
import warnings
warnings.filterwarnings("ignore")


org_out = './upload/'
device_1 = 'cuda:0'
device_2 = 'cuda:0'
detection = DETECT(weight_path="./weights/craft_v1.pth", device=device_1)
# detection = DETECT(device=device_2)
recognition = RECOGNIZE(device=device_2)
temp_classify = TEMP_CLASSIFY(weight_path='./weights/template_classify_mbn.pth', device='cpu')

# gcn_chungnhan = INFO_EXTRACT(weight_path='./weights/layout_v2_final_gcn_chungnhan_2022-11-27.pth', template='gcn_chungnhan', device=device_2)
# gcn_nhao_dato = INFO_EXTRACT(weight_path='./weights/layout_v2_final_gcn_nhao_dato_2023-01-17.pth', template='gcn_nhao_dato', device=device_1)
# gcn_qsdd_mst = INFO_EXTRACT(weight_path='./weights/layout_v2_final_gcn_qsdd_mst_2023-06-23.pth', template='gcn_qsdd_mst', device=device_2)
# gcn_qsdd_mt = INFO_EXTRACT(weight_path='./weights/layout_v2_final_gcn_qsdd_mt_2023-01-20.pth', template='gcn_qsdd_mt', device=device_2)
# gcn_qsdd_msp = INFO_EXTRACT(weight_path ='./weights/layout_v2_final_gcn_qsdd_msp_2022-11-27.pth', template='gcn_qsdd_msp', device=device_1)
# gcn_qsdd_tsgl_mc = INFO_EXTRACT(weight_path ='./weights/layout_v2_final_gcn_qsdd_tsgl_mc_2022-11-27.pth', template='gcn_qsdd_tsgl_mc', device=device_2)
# gcn_qsdd_tsgl_ms = INFO_EXTRACT(weight_path ='./weights/layout_v2_final_gcn_qsdd_tsgl_ms_2023-05-21.pth', template='gcn_qsdd_tsgl_ms', device=device_2)
# gcn_qsdd_tsgl_mt = INFO_EXTRACT(weight_path ='./weights/layout_v2_final_gcn_qsdd_tsgl_mt_2023-02-07.pth', template='gcn_qsdd_tsgl_mt', device=device_2)
# gcn_qsdd_tsgl_ms = INFO_EXTRACT(weight_path ='./weights/layout_v2_final_gcn_qsdd_tsgl_ms_2023-05-21.pth', template='gcn_qsdd_tsgl_ms', device=device_1)

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

org_input_path = './Data/new_splited/gcn_qsdd_tsgl_mt'
org_output_path = './Data/predict/'
for img_file in os.listdir(os.path.join(org_input_path)):
    image = Image.open(os.path.join(org_input_path,img_file))
    image = image.convert("RGB")
    image = image.resize((int(1000*image.size[0]/image.size[1]), 1000))
    img = np.array(image)
    begin = datetime.now()
    print('===== start =====')
    try:
        info_extract, template_type = get_model_extract(image)
    except:
        continue

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
        
    if template_type in ['gcn_chungnhan', 'gcn_nhao_dato', 'gcn_qsdd_mst', 'gcn_qsdd_tsgl_mt']:
        informations = convert_to_same_with_csh(info)
    else:
        informations = convert_to_same_no_csh(info)
    # check template is exist in output path
    if not os.path.exists(os.path.join(org_output_path, template_type)):
        os.mkdir(os.path.join(org_output_path, template_type))
    # check type_data is exist in output path
    if not os.path.exists(os.path.join(org_output_path, template_type)):
        os.mkdir(os.path.join(org_output_path, template_type))
    output_path = os.path.join(org_output_path, template_type, img_file.split('.')[0])
    print(output_path)
    with open(f'{output_path}.txt', 'w', encoding='utf-8') as f:
        for item in informations:
            label = item['label']
            sentences = item['sentences']
            # Nếu câu thông tin là một chuỗi
            if isinstance(sentences, str):
                f.write(f"{label}|{sentences}\n")
            # Nếu câu thông tin là một danh sách
            elif isinstance(sentences, list):
                sentences_str = "|".join(sentences)
                f.write(f"{label}|{sentences_str}\n")
    print('kie and draw and post time', datetime.now() - start_time)
    total = datetime.now() - begin
    print('total time consuming', datetime.now() - begin)
    print('===== done =====')
    
