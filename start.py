from mvector.utils.utils import dict_to_object
from ruamel import yaml

from loadTXT import loadTXT
from train import train
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
        train(config=config.train)
    if config.start.summary:
        modelSummary(config=config.summary)
    pass
