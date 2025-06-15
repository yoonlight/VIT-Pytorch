import yaml
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary
from torchvision import models

def print_model_summary(model, input_size=(3, 224, 224)):
    model.to(device)
    model.eval()
    summary(model, input_size=input_size)

if __name__ == "__main__":
    # -- (생략) config 로딩, 시드 고정 등 기존 로직 그대로 --
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config_path',
                        default='config/default.yaml', type=str)
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 시드 고정
    seed = config['train_params']['seed']
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) torchvision에서 pretrained MobileNetV2 불러오기
    #    (torchvision>=0.13이면 weights 인자를 권장합니다)
    # model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model = models.mobilenet_v2()

    # 2) classification head 교체 (1000→10)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, config['model_params']['num_classes'])

    # 3) 요약 출력
    input_size = (
        config['model_params']['im_channels'],
        config['model_params']['image_height'],
        config['model_params']['image_width']
    )
    print_model_summary(model, input_size=input_size)
