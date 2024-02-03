from mvector.utils.utils import dict_to_object
from ruamel import yaml

from eval_LFW import evalLFW
from loadTXT import loadTXT
from speedTest import speedTest
from train import train_facenet, train_arcface
from summary import modelSummary

# import matplotlib
# matplotlib.use('TkAgg')
if __name__ == "__main__":
    with open('config.yml', 'r', encoding='utf-8') as f:
        configs = yaml.load(f.read(), Loader=yaml.RoundTripLoader)
    config = dict_to_object(configs)
    if config.start.reloadData:
        loadTXT(config=config.dataset)
    if config.start.startTrain:
        if config.train.model == 'facenet':
            train_facenet(config=config.train)
        elif config.train.model == 'arcface':
            train_arcface(config=config.train)
        else:
            raise ValueError('model error')
    if config.start.summary:
        modelSummary(config=config.summary)
    if config.start.speed:
        speedTest(config=config.speed)
    if config.start.evalLFW:
        evalLFW(config=config.LFW)
    pass
