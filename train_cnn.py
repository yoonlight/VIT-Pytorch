import yaml
import argparse
import torch
import random
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# 추가된 import
import torch.nn as nn
from torchvision import models

from dataset.mnist_color_texture_dataset import MnistDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_for_one_epoch(epoch_idx, model, mnist_loader, optimizer):
    losses = []
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for data in tqdm(mnist_loader, desc=f"Epoch {epoch_idx+1}"):
        im = data['image'].float().to(device)
        number_cls = data['number_cls'].long().to(device)

        optimizer.zero_grad()
        outputs = model(im)
        loss = criterion(outputs, number_cls)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    avg_loss = np.mean(losses)
    print(f'Finished epoch: {epoch_idx+1} | Number Loss : {avg_loss:.4f}',
          file=open(os.path.join('train_log.txt'), 'a'))
    return avg_loss


def train(args):
    # 1) config file 읽기
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)
    print(config)

    # 2) 시드 고정
    seed = config['train_params']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

    # 3) 모델 생성: 학습되지 않은 MobileNetV2
    model = models.mobilenet_v2()
    # classifier head 교체 (1000 -> num_classes)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(
        in_features, config['model_params']['num_classes'])
    model = model.to(device)

    # 4) 데이터셋 & DataLoader
    mnist = MnistDataset(
        split='train',
        config=config['dataset_params'],
        im_h=config['model_params']['image_height'],
        im_w=config['model_params']['image_width']
    )
    mnist_loader = DataLoader(
        mnist,
        batch_size=config['train_params']['batch_size'],
        shuffle=True,
        num_workers=4
    )

    # 5) 옵티마이저 & 스케줄러
    optimizer = Adam(model.parameters(), lr=config['train_params']['lr'])
    scheduler = ReduceLROnPlateau(
        optimizer, factor=0.5, patience=2, verbose=True)

    # 6) 체크포인트 디렉토리 생성
    task_dir = config['train_params']['task_name']
    if not os.path.exists(task_dir):
        os.makedirs(task_dir, exist_ok=True)

    # 7) 체크포인트 로드 (있다면)
    ckpt_path = os.path.join(task_dir, config['train_params']['ckpt_name'])
    if os.path.isfile(ckpt_path):
        print('Loading checkpoint')
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # 8) 학습 루프
    best_loss = np.inf
    for epoch_idx in range(config['train_params']['epochs']):
        mean_loss = train_for_one_epoch(
            epoch_idx, model, mnist_loader, optimizer)
        scheduler.step(mean_loss)

        if mean_loss < best_loss:
            print(f'Improved Loss to {mean_loss:.4f} …. Saving Model')
            torch.save(model.state_dict(), ckpt_path)
            best_loss = mean_loss
        else:
            print('No Loss Improvement')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Arguments for training MobileNetV2 on MNIST')
    parser.add_argument('--config', dest='config_path',
                        default='config/mobilenet.yaml', type=str)
    args = parser.parse_args()
    train(args)
