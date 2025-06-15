import yaml
import random
import argparse
import numpy as np
import torch
from torchsummary import summary


def print_model_summary(model, input_size=(3, 32, 32)):
    """
    Print the summary of the model.
    :param model: The model to summarize.
    :param input_size: The size of the input tensor.
    """
    model.to('cuda')
    summary(model, input_size=input_size)


if __name__ == "__main__":
    from model.transformer import VIT
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser(description='Arguments for vit training')
    parser.add_argument('--config', dest='config_path',
                        default='config/default.yaml', type=str)
    args = parser.parse_args()
    # Create a VIT model instance with default parameters
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    seed = config['train_params']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    vit_model = VIT(config['model_params']).to(device)

    input_size = (config['model_params']['im_channels'],
                  config['model_params']['image_height'],
                  config['model_params']['image_width'])

    # Print the model summary
    print_model_summary(vit_model, input_size=input_size)
