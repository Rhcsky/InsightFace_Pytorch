import sys
sys.path.append("/home/user/Project/ocr/InsightFace_Pytorch")
from Insightface import Insightface, get_config

def recognize(img_dir):
    conf = get_config()
    insight = Insightface()
    name, frame = insight.inference(conf, img_dir)

    return name, frame