import os
from pathlib import Path

import torch
from torch.nn import CrossEntropyLoss
from torchvision import transforms as trans

from multiprocessing import Process, Pipe,Value,Array
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank, logging_time

import cv2
from PIL import Image
from tqdm import tqdm
from glob import glob
from easydict import EasyDict as edict

class Insightface():
    def __init__(self):
        pass

    def train(self,conf):
        print(f'Train Start. Train dataset is {conf.data_mode}')
        learner = face_learner(conf,False)
        learner.train(conf, conf.epoch)

    def inference(self,conf,img):
        mtcnn = MTCNN()
        learner = face_learner(conf,True)
        learner.load_state(conf,'final.pth',True,True)
        learner.model.eval()
        targets, names = load_facebank(conf)
        
        image = Image.open(img)
        frame = cv2.imread(img,cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            bboxes, faces = mtcnn.align_multi(image, conf.face_limit, conf.min_face_size)
            bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
            bboxes = bboxes.astype(int)
            bboxes = bboxes + [-1,-1,1,1] # personal choice    
            results, score = learner.infer(conf, faces, targets, False)
            name = names[results[0]+1]
            frame = draw_box_name(bboxes[0], name, frame)
        except Exception as ex:
            name = "Can't detect face."
            h, w, c = frame.shape
            bbox = [int(h*0.5),int(w*0.5),int(h*0.5),int(w*0.5)]
            frame = draw_box_name(bbox, name, frame)
            
        return name, frame

    def update_facebank(self,conf):
        mtcnn = MTCNN()
        learner = face_learner(conf,True)
        learner.load_state(conf,'final.pth',True,True)
        learner.model.eval()
        _, _ = prepare_facebank(conf,learner.model,mtcnn,False)
    
    def test(self,conf,img_dir,update=False,view_score=False,view_error=False):
        #Load models
        mtcnn = MTCNN()
        learner = face_learner(conf, True)
        if conf.device.type == 'cpu':
            learner.load_state(conf,'cpu_final.pth',True,True)
        else:
            learner.load_state(conf,'final.pth',True,True)
        learner.model.eval()

        #Load Facebank
        if update:
            targets, names = prepare_facebank(conf, learner.model, mtcnn, False)
            print('facebank updated')
        else:
            targets, names = load_facebank(conf)
            print('facebank loaded')

        #Load Image list
        img_list = glob(img_dir + '**/*.jpg')
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
                results, score = learner.infer(conf, faces, targets, False)
                for idx,bbox in enumerate(bboxes):
                    print(f'{Label}: {score[idx]}')
                    if view_score:
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

            f = len(bboxes)
            tf = str(True if label in preds else False)
            t = f'{f:^15}{label:^20}{tf:^15}{acc/(i+1):^15.4}'
            pbar.set_description(t)
        
        if detect_err>0:
            print(f'Detect Error: {detect_err}')
            if view_error:
                pp(fails)
            else:
                print(f'If you want to see details, make veiw_error True.')

        print(f'Accuracy: {acc/len(img_list)}')


def get_config(train = False, facebank='facebank'):
    conf = edict()
    conf.data_path = Path('/home/user/Project/ocr/InsightFace_Pytorch/data/')
    conf.work_path = Path('/home/user/Project/ocr/InsightFace_Pytorch/work_space/')
    conf.model_path = conf.work_path/'models'
    conf.log_path = conf.work_path/'log'
    conf.save_path = conf.work_path/'save'
    conf.input_size = [112,112]
    conf.embedding_size = 512
    conf.use_mobilfacenet = False
    conf.net_depth = 50
    conf.drop_ratio = 0.6
    conf.net_mode = 'ir_se' # or 'ir'
    conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf.test_transform = trans.Compose([
                    trans.ToTensor(),
                    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
    conf.data_mode = 'emore'
    conf.vgg_folder = conf.data_path/'faces_vgg_112x112'
    conf.ms1m_folder = conf.data_path/'faces_ms1m_112x112'
    conf.emore_folder = conf.data_path/'faces_emore'
    conf.batch_size = 100 # irse net depth 50 

#--------------------Training Config ------------------------    
    if train:
        conf.log_path = conf.work_path/'log'
        conf.save_path = conf.work_path/'save'
        conf.lr = 0.1
        conf.milestones = [12,15,18]
        conf.momentum = 0.9
        conf.pin_memory = True
        conf.num_workers = 3
        conf.ce_loss = CrossEntropyLoss()    
        conf.epoch = 20
#--------------------Inference Config ------------------------
    else:
        conf.facebank_path = conf.data_path/facebank
        conf.threshold = 0.9
        conf.face_limit = 5
        conf.min_face_size = 50

    return conf
