import torch
import torch.backends.cudnn as cudnn

from nets.arcface import Arcface
from nets.facenet import Facenet
from utils.dataloader import LFWDataset
from utils.evaluate import startEval


def evalLFW(config):
    test_loader = torch.utils.data.DataLoader(
        LFWDataset(dir=config.dirPath, pairs_path=config.pairPath, image_size=config.inputSize), batch_size=config.batchSize,
        shuffle=False)
    if config.model == 'facenet':
        model = Facenet(backbone=config.backbone, mode="predict")
    elif config.model == 'arcface':
        model = Arcface(backbone=config.backbone, mode="predict")
    else:
        raise ValueError('unsupported model')
    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.eval()
    model.load_state_dict(torch.load(config.weightPath, map_location=device), strict=False)
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    model = model.cuda()
    startEval(test_loader, model, config.pngPath, config.logInterval, config.batchSize)
