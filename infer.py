import cv2
from PIL import Image
import argparse
from pathlib import Path
from multiprocessing import Process, Pipe,Value,Array
import torch
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank, logging_time
import os
from pprint import pprint as pp
from tqdm import tqdm
from glob import glob

import time

@logging_time
def load_MTCNN():
    return MTCNN()

@logging_time
def load_Learner(conf,args):
    learner = face_learner(conf, True)
    learner.threshold = args.threshold
    if conf.device.type == 'cpu':
        learner.load_state(conf, 'cpu_final.pth', True, True)
    else:
        learner.load_state(conf, 'final.pth', True, True)
    learner.model.eval()
    return learner

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-s", "--save", help="whether save",action="store_true")
    parser.add_argument('-th','--threshold',help='threshold to decide identical faces',default=0.9, type=float)
    parser.add_argument("-u", "--update", help="whether perform update the facebank",action="store_true")
    parser.add_argument("-tta", "--tta", help="whether test time augmentation",action="store_true")
    parser.add_argument("-c", "--score", help="whether show the confidence score",action="store_true")
    parser.add_argument("-d", "--image_dir", type=str, help="dir of verification picture", required=True)
    parser.add_argument("-f", "--facebank", type=str, help="dir of facebank picture", default='facebank')
    
    args = parser.parse_args()

    conf = get_config(False)
    mtcnn = load_MTCNN()
    learner = load_Learner(conf,args)
    conf.facebank_path = conf.data_path/args.facebank

    start = time.time()
    if args.update:
        targets, names = prepare_facebank(conf, learner.model, mtcnn, False)
        print('facebank updated')
    else:
        targets, names = load_facebank(conf)
        print('facebank loaded')
    print(f'WorkingTime[Load facebank]: {time.time()-start}')

    print("Names")

    # list image directory
    img_list = glob(args.image_dir + '**/*.jpg')
    acc = 0
    detect_err=0
    fails = []
    print(f"{'Found':^15}{'Name':^20}{'Result':^15}{'Score':^15}")

    pbar = enumerate(img_list)
    pbar = tqdm(pbar, total = len(img_list))
    for i, x in pbar:
        preds = []
        label = str(os.path.dirname(x))
        label = os.path.basename(label)
        image = Image.open(x)
        frame = cv2.imread(x,cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            bboxes, faces = mtcnn.align_multi(image, conf.face_limit, conf.min_face_size)
            bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
            bboxes = bboxes.astype(int)
            bboxes = bboxes + [-1,-1,1,1] # personal choice    
            results, score = learner.infer(conf, faces, targets, args.tta)
            for idx,bbox in enumerate(bboxes):
                # print(f'{label}-{names[results[idx]+1]}: {score[idx]}')
                if args.score:
                    frame = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), frame)
                else:
                    frame = draw_box_name(bbox, names[results[idx] + 1], frame)
                preds.append(names[results[idx]+1])

            if label in preds:
                acc += 1
            else:
                fails.append([label,preds])
                # Image.fromarray(frame,'RGB').show()
        except Exception as ex:
            fails.append([label,ex])
            detect_err += 1
            # print(f'detect error')

        f = len(bboxes)
        tf = str(True if label in preds else False)
        t = f'{f:^15}{label:^20}{tf:^15}{acc/(i+1):^15.4}'
        pbar.set_description(t)
    # if len(fails)>0:
    #     print('='*15 + 'Fail list' + '='*15)
    #     print(f'    Label\tPred')
    #     pp(fails)
    #     if detect_err:
    #         print(f'Detect error = {detect_err}')
    #         print(f'Accuract: {acc/(len(img_list)-detect_err)}')

    print(f'Accuracy: {acc/len(img_list)}')
