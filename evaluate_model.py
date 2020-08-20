import pdb

from config import get_config
from Learner import face_learner
from data.data_pipe import get_val_pair
from torchvision import transforms as trans

conf = get_config(training=False)
learner = face_learner(conf,inference=True)

learner.load_state(conf,'final.pth',model_only=True, from_save_folder = True)

lfw, lfw_issame = get_val_pair(conf.emore_folder, 'lfw')

accuracy, best_threshold, roc_curve_tensor = learner.evaluate(conf, lfw, lfw_issame, nrof_folds=10, tta=False)
print('lfw - accuray:{}, threshold:{}'.format(accuracy, best_threshold))
trans.ToPILImage()(roc_curve_tensor)

