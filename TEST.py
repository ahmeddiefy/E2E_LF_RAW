import numpy as np
import cv2
from scipy import misc
from PIL import Image
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import glob, os, re
from PSNR import psnr
import scipy.io
import pickle
from MODEL import model
import time
from scipy.io import savemat
DATA_PATH = "./data/test/"

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_path")
args = parser.parse_args()
model_path = args.model_path
def get_img_list(data_path):
    l = glob.glob(os.path.join(data_path,"*"))
    l = [f for f in l if re.search("^\d+.mat$", os.path.basename(f))]
    train_list = []
    for f in l:
        if os.path.exists(f):
            if os.path.exists(f[:-4]+"_1.mat"): train_list.append([f, f[:-4]+"_1.mat", 2])

    return train_list
def get_test_image(test_list, offset, batch_size):
    target_list = test_list[offset:offset+batch_size]
    input_list = []
    gt_list = []
    scale_list = []
    for pair in target_list:
        mat_dict = scipy.io.loadmat(pair[1])
        input_img = None
        if ("img_1") in mat_dict: 	input_img = mat_dict["img_1"]
        else: continue
        gt_img = scipy.io.loadmat(pair[0])['img_raw']
        input_list.append(input_img)
        gt_list.append(gt_img)
        scale_list.append(pair[2])
    return input_list, gt_list, scale_list
def test_LF_E2E_with_sess(epoch, ckpt_path, data_path,sess):
    folder_list = glob.glob(os.path.join(data_path, 'Set*'))
    # saver.restore(sess, ckpt_path)
    saver.restore(sess, tf.train.latest_checkpoint('D:/LF_E2E/checkpoints/'))
    psnr_dict = {}
    for folder_path in folder_list:
        psnr_list = []
        img_list = sorted(get_img_list(folder_path))
        for i in range(len(img_list)):
            img_path = img_list[i]
            img_name = img_path[0].split('x')[1].split('.mat')[0]
            input_list, gt_list, scale_list = get_test_image(img_list, i, 1)
            input_y = input_list[0]
            gt_y = gt_list[0]
            start_t = time.time()
            img_vdsr_y = sess.run([output_tensor], feed_dict={input_tensor: np.resize(input_y, (1, input_y.shape[0], input_y.shape[1], input_y.shape[2]))})
            img_vdsr_y = np.resize(img_vdsr_y, (gt_y.shape[0], gt_y.shape[1]))
            end_t = time.time()
            print("end_t",end_t,"start_t",start_t)
            print("time consumption",end_t-start_t)
            print("image_size", input_y.shape)
            print('Max:  ', np.amax(img_vdsr_y))
            print('Min:  ',np.amin(img_vdsr_y))

            
            save_path = 'D:/LF_E2E/results/numpy/'+img_name[1:]+'_gt.mat'
            scipy.io.savemat(save_path,{"a": gt_y, "label": "experiment"})
            
            save_path = 'D:/LF_E2E/results/numpy/'+img_name[1:]+'.mat'
            scipy.io.savemat(save_path,{"a": img_vdsr_y, "label": "experiment"})
            

           
            


            
        psnr_dict[os.path.basename(folder_path)] = psnr_list
    with open('psnr/%s' % os.path.basename(ckpt_path), 'wb') as f:
        pickle.dump(psnr_dict, f)
def test_LF_E2E(epoch, ckpt_path, data_path):
	with tf.Session() as sess:
		test_LF_E2E_with_sess(epoch, ckpt_path, data_path, sess)
if __name__ == '__main__':
    model_list = sorted(glob.glob("./checkpoints/LF_E2E_adam_epoch_*"))
    print('model_list', model_list)
    model_list = [fn for fn in model_list if not os.path.basename(fn).endswith("meta")]
    model_list = [fn for fn in model_list if not os.path.basename(fn).endswith("index")]
    with tf.Session() as sess:
        input_tensor = tf.placeholder(tf.float32, shape=(1, None, None, 4))
        shared_model = tf.make_template('shared_model', model)
        output_tensor, weights 	= shared_model(input_tensor)
        saver = tf.compat.v1.train.Saver(weights)
        tf.initialize_all_variables().run()
        for model_ckpt in model_list:
            print(model_ckpt)
            epoch = int(model_ckpt.split('epoch_')[-1].split('.ckpt')[0])
            print(epoch)
			#if epoch<60:
			#	continue
            test_LF_E2E_with_sess(epoch, model_ckpt, DATA_PATH,sess)
