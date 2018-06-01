"""
Demo of HMR.

Sample usage:
python -m demo_video --video hoge.mp4
"""

from absl import flags
import cv2
import numpy as np
import os
import sys
import time

import skimage.io as io
import tensorflow as tf

from src.util import renderer as vis_util
from src.util import image as img_util
from src.util import openpose as op_util
import src.config
from src.RunModel import RunModel

flags.DEFINE_string('video', '', 'Video to run')

def visualize(img, proc_param, joints, verts, cam):
    cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
        proc_param, verts, cam, joints, img_size=img.shape[:2])

    # Render results
    skel_img = vis_util.draw_skeleton(img, joints_orig)
    return skel_img


def preprocess_image(img, box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]
    scale = 150. / float(y2 - y1)
    center_pos = ((x1 + x2) / 2, (y1 + y2) / 2)
    center = np.round(np.array(center_pos)).astype(int)
    crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                               config.img_size)
    # Normalize image to [-1, 1]
    crop = 2 * ((crop / 255.) - 0.5)
    return crop, proc_param, img

def main(video):
    sess = tf.Session()
    model = RunModel(config, sess=sess)

    cap = cv2.VideoCapture(video)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    CV_CAP_PROP_FRAME_COUNT = 7
    frame_count = int(cap.get(CV_CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    file_step = 150

    out = False

    out_path = "out"
    frame_num = 0

    start = time.time()

    while(cap.isOpened()):
        frame_num += 1

        if frame_num % 10 == 1:
            print("{} / {} ({}%) {}fps".format(
                frame_num, 
                frame_count, 
                int(100 * float(frame_num) / frame_count), 
                frame_num / (0.0001 + time.time() - start)))

        if frame_num % file_step == 1:
            if out:
                out.release()
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            filepath = "{}/{}_{}.mp4".format(out_path, frame_num, frame_num + file_step - 1)
            out = cv2.VideoWriter(filepath, fourcc, int(fps), (width, height))

        ret, img_raw = cap.read()
        if not ret:
            break

        img_raw_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

        bounding_boxes = [[0, 0, width, height]]

        rendered_img = None
        for box in bounding_boxes:
            input_img, proc_param, img = preprocess_image(img_raw_rgb, box)

            if rendered_img is None:
                rendered_img = img

            input_img = np.expand_dims(input_img, 0)

            joints, verts, cams, joints3d, theta = model.predict(
                input_img, get_theta=True)

            rendered_img = visualize(rendered_img, proc_param, joints[0], verts[0], cams[0])
            rendered_img = cv2.cvtColor(rendered_img, cv2.COLOR_RGBA2RGB)

        out_img = cv2.cvtColor(rendered_img, cv2.COLOR_RGB2BGR)
        out.write(out_img)

    cap.release()
    out.release()

if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    # Using pre-trained model, change this to use your own.
    config.load_path = src.config.PRETRAINED_MODEL
    config.batch_size = 1
    main(config.video)
