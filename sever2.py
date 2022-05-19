# coding:utf-8
import skimage
from flask import Flask,render_template,request,redirect,url_for
from werkzeug.utils import secure_filename
import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import compute_score
from config import Config
# import coco
import utils
import model as modellib
import visualize

import torch

app = Flask(__name__)


def predict(file_names):
    # Root directory of the project
    ROOT_DIR = os.getcwd()

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    # Path to trained weights file
    # Download this file and place in the root of your
    # project (See README file for details)
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_shapes_0005.pth")


    # Directory of images to run detection on
    IMAGE_DIR = os.path.join(ROOT_DIR, "images")
    class ShapesConfig(Config):
        """Configuration for training on the toy shapes dataset.
        Derives from the base Config class and overrides values specific
        to the toy shapes dataset.
        """
        # Give the configuration a recognizable name
        NAME = "shapes"

        # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
        # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
        GPU_COUNT = 1
        IMAGES_PER_GPU = 4

        # Number of classes (including background)
        NUM_CLASSES = 1 + 6  # background + 3 shapes

        # Use small images for faster training. Set the limits of the small side
        # the large side, and that determines the image shape.
        IMAGE_MIN_DIM = 100
        IMAGE_MAX_DIM = 448
        # Use smaller anchors because our image and objects are small
        RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels

        # Reduce training ROIs per image because the images are small and have
        # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
        TRAIN_ROIS_PER_IMAGE = 100

        # Use a small epoch since the data is simple
        STEPS_PER_EPOCH = 100

        # use small validation steps since the epoch is small
        VALIDATION_STEPS = 50

    class InferenceConfig(ShapesConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        # GPU_COUNT = 0 for CPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()
    # Create model object.
    model = modellib.MaskRCNN(model_dir=MODEL_DIR, config=config)
    if config.GPU_COUNT:
        model = model.cuda()
    # Load weights trained on MS-COCO
    model.load_state_dict(torch.load(COCO_MODEL_PATH))
    # COCO Class names
    # Index of the class in the list is its ID. For example, to get ID of
    # the teddy bear class, use: class_names.index('teddy bear')
    class_names = ['BG', "fastener_L1", "fastener_R1", "hat_L", "hat_R", "shim_L", "shim_R"]
    image = skimage.io.imread(file_names)

    # Run detection
    results = model.detect([image])

    # Visualize results
    r = results[0]
    fl1_point, fl2_point, hl_point, hr_point, sl_rect, sr_rect, contour_left, contour_right = visualize.get_fastener_polygon(
        image, r['rois'], r['masks'], r['class_ids'],
        class_names, r['scores'])
    total_left_score, total_left_score_result, total_right_score, total_right_score_result, total_hat_score, total_shim_score = compute_score.cpmpute_score_all(
        fl1_point, fl2_point, hl_point, hr_point, sl_rect, sr_rect, contour_left, contour_right)
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                          class_names, r['scores'])
    save_path = os.path.join('./',file_names)
    plt.savefig(save_path)
    return total_left_score, total_left_score_result, total_right_score, total_right_score_result, total_hat_score, total_shim_score
@app.route('/')
def index_page():
    return render_template('index.html')
def return_img_stream(img_local_path):
    """
    工具函数:
    获取本地图片流
    :param img_local_path:文件单张图片的本地绝对路径
    :return: 图片流
    """
    import base64
    img_stream = ''
    with open(img_local_path, 'rb') as img_f:
        img_stream = img_f.read()
        img_stream = base64.b64encode(img_stream).decode()
    return img_stream

@app.route('/upload_image', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['input_image']
        imagefilename = f.filename
        if f:
            f_dirpath = './resources/received_images'
            if not os.path.isdir(f_dirpath):
                os.makedirs(f_dirpath)
            imagefilepath = os.path.join(f_dirpath,imagefilename)
            f.save(imagefilepath)

            print('图片文件保存地址：%s'%imagefilepath)
            total_left_score, total_left_score_result, total_right_score, total_right_score_result, total_hat_score, total_shim_score = predict(imagefilepath)
            total_left_score= str('1️⃣ 左侧扣件评分为：'+str(total_left_score * 100)[0:6]+ '分，左侧扣件同学再接再厉呢！')
            total_left_score_result = str('2️⃣ 左侧扣件鉴定为：'+str(total_left_score_result))
            total_right_score = str('3️⃣ 右侧扣件评分为：'+str(total_right_score * 100)[0:6]+'分，右侧扣件同学也不错呢！')
            total_right_score_result = str('4️⃣ 右侧扣件鉴定为：'+str(total_right_score_result))
            total_hat_score_fast = str('5️⃣ 两侧扣件是否扣紧：'+'两侧扣件质点距离为'+str(total_hat_score[3])[0:6]+'，他们相隔约为'+str(total_hat_score[3] / 37)[0:4]+'个扣件帽长度，计算可得'+str(total_hat_score[0]))
            total_hat_score_theta = str('6️⃣ 其中'+str(total_hat_score[1])+'，'+str(total_hat_score[2]))
            total_hat_score_sim = str('7️⃣ 通过HU矩计算相似度可得：'+str(total_hat_score[4])+'，'+str(total_hat_score[5]))
            total_shim_score = str('8️⃣ 本系统通过旋转程度来计算两侧垫片是否拧紧，其中'+str(total_shim_score[0])+'，'+str(total_shim_score[1])+'；而后通过HU矩计算相似度来度量完整程度，其中'+str(total_shim_score[2])+'，'+str(total_shim_score[3]))
            img_stream = return_img_stream(imagefilepath)
            # total_shim_score = str(total_shim_score)
            # total_shim_score_sim = str('8️⃣ 两侧垫片是否拧紧：'+str(total_shim_score[1])+'，'+str(total_shim_score[2]))
            return render_template('result.html',total_left_score = total_left_score,total_left_score_result = total_left_score_result,
                                   total_right_score = total_right_score , total_right_score_result =  total_right_score_result,
                                   total_hat_score_fast = total_hat_score_fast ,
                                   total_hat_score_theta = total_hat_score_theta ,total_hat_score_sim = total_hat_score_sim,
                                   total_shim_score =total_shim_score,img_stream=img_stream)
            # return total_left_score, total_left_score_result, total_right_score, total_right_score_result, total_hat_score, total_shim_score

if __name__ == '__main__':
    app.run(port=6006,host='0.0.0.0',debug=True)