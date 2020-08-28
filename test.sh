python

from Insightface import Insightface, get_config
conf = get_config()
ins = Insightface()
name, frame = ins.inference(conf,'data/one/1.jpg')
# ins.test(conf,'data/lfw_funneled/')

print(name)