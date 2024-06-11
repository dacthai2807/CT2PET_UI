from flask import Flask, render_template, request, flash, redirect, jsonify
from flask_cors import CORS, cross_origin
from segmentation_lbbdm import infer as seg_lbbdm_infer
from baseline import infer as baseline_infer
import os 
from PIL import Image
import pydicom
import torchvision.transforms as transforms
from utils import create_to_gen_html

app = Flask(__name__)
app.secret_key = "super secret key"

cors = CORS(app, resource={
    r"/*":{
        "origins":"*"
    }
}) 

@app.route('/', methods=['GET', 'POST'])
def get_info():
    if request.method == "GET":
        return render_template('form_image1.html', file='')
    if request.method == "POST":
        uploaded_file = request.files["formFile"]
        dicom_data = pydicom.dcmread(uploaded_file, force=True)
        pixel_data = dicom_data.pixel_array 

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        np_ct_image = pixel_data / float(2047) # normalize 
        ct_image = Image.fromarray(np_ct_image) 

        x_cond = transform(ct_image) 

        img_str1 = seg_lbbdm_infer(x_cond)
        img_str2 = baseline_infer(x_cond)
        
        img_str1 = create_to_gen_html(img_str1, 'Ảnh PET của mô hình đề xuất')
        img_str2 = create_to_gen_html(img_str2, 'Ảnh PET của mô hình hiện tại')
        ct = create_to_gen_html(pixel_data, 'Ảnh CT ban đầu')

        # print(pixel_data.shape)

        # return render_template('form_image1.html', file='Image Upload Succeed', img_data=img_str)
        return render_template('form_image1.html', file='Image Upload Succeed', img_data=ct ,img_data1=img_str1, img_data2=img_str2)

@app.route('/get_ct_pixel_data', methods=['POST'])
@cross_origin()
def get_ct_pixel_data():
    try:
        uploaded_file = request.files["file"]
        # print(uploaded_file)
        dicom_data = pydicom.dcmread(uploaded_file, force=True)
        pixel_data = dicom_data.pixel_array 
        
        ct = create_to_gen_html(pixel_data, None)
    
        result = {
            "data": {
                "ct": ct,
            }
        }

        return jsonify(result)
    except:
        return jsonify({"data": "Invalid ct format"}), 404

import json    
import random
import numpy as np

CT_PET_PATH = '/mnt/disk1/PET_CT/combined'
ELEMENTS_FILE_PATH = '/mnt/disk1/PET_CT/records/elements.json'

def load_elements():
    if os.path.exists(ELEMENTS_FILE_PATH):
        with open(ELEMENTS_FILE_PATH, 'r') as file:
            return json.load(file)
    else:
        new_elements = [(a, b) for a in range(3) for b in range(2688)]
        save_elements(new_elements)
        return new_elements

def save_elements(elements):
    with open(ELEMENTS_FILE_PATH, 'w') as file:
        json.dump(elements, file)
    
@app.route('/get_ct_pet', methods=['GET'])
@cross_origin()
def get_ct_pet():
    elements = load_elements()
    
    print(len(elements))
    
    if not elements:
        return jsonify({"data": "No more ct & pet"}), 404
    
    try:
        case, img_id = random.choice(elements)    
        
        ct_img = np.load(os.path.join(CT_PET_PATH, 'A', f'{img_id}.npy'), allow_pickle=True)
        
        if case != 2:
            pet_img = np.load(os.path.join(CT_PET_PATH, 'B', f'{img_id}.npy'), allow_pickle=True)
        else:
            pet_img = np.load(os.path.join(CT_PET_PATH, 'C', f'{img_id}.npy'), allow_pickle=True)
        
        ct = create_to_gen_html(ct_img, None)
        pet = create_to_gen_html(pet_img, None)
    
        result = {
            "data": {
                "img_id": img_id,
                "case": case,
                "ct": ct,
                "pet": pet
            }
        }

        return jsonify(result)
    except:
        return jsonify({"data": "Cannot get ct & pet"}), 404

import csv
SAVE_PATH = '/mnt/disk1/PET_CT/records/patients.csv'

def write_to_csv(json_data):
    file_exists = os.path.exists(SAVE_PATH)

    with open(SAVE_PATH, 'a', newline='') as file:
        fieldnames = ['img_id', 'case', 'status', 'description']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'img_id': json_data['img_id'],
            'case': json_data['case'],
            'status': json_data.get('status', ''), 
            'description': json_data.get('description', '')  
        })

@app.route('/post_ct_pet', methods=['POST'])
@cross_origin()
def post_ct_pet():
    elements = load_elements()
    
    if not elements:
        return jsonify({"data": "No more ct & pet"}), 404
    
    try:
        json_data = request.json    
        elements.remove([json_data['case'], json_data['img_id']])
        save_elements(elements)
        write_to_csv(json_data)

        return jsonify({"data": "Posted successfully!"})
    except:
       return jsonify({"data": "Cannot post ct & pet"}), 404

@app.route('/upload_ct', methods=['POST'])
@cross_origin()
def upload_ct():
    try:
        uploaded_file = request.files["file"]
        # print(uploaded_file)
        dicom_data = pydicom.dcmread(uploaded_file, force=True)
        pixel_data = dicom_data.pixel_array 

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        np_ct_image = pixel_data / float(2047) # normalize 
        ct_image = Image.fromarray(np_ct_image) 

        x_cond = transform(ct_image) 

        img_str1 = seg_lbbdm_infer(x_cond)
        img_str2 = baseline_infer(x_cond)
        
        img_str1 = create_to_gen_html(img_str1, None)
        img_str2 = create_to_gen_html(img_str2, None)
        ct = create_to_gen_html(pixel_data, None)
    
        result = {
            "data": {
                "ct": ct,
                "proposed_pet": img_str1,
                "baseline_pet": img_str2
            }
        }

        return jsonify(result)
    except:
        return jsonify({"data": "Cannot translate this file"}), 404
    

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 9999))
    app.run(debug=False, host="0.0.0.0", port=port)
