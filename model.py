import torch
from torch import nn
from models import mobilenet, mlp


def generate_model(args, input_size):
    assert args.model_type in ['resnet', 'shufflenet', 'shufflenetv2', 'mobilenet', 'mobilenetv2', 'mlp']

    if args.model_type == 'mobilenet':
        model = mobilenet.get_model(
            sample_size=args.sample_size,
            width_mult=args.width_mult,
            pre_train=args.pre_train_model
        )
    elif args.model_type == 'mlp':
        # model = mlp.get_model(input_size=121, layer1_size=128, layer2_size=256, output_size=512)
        model = mlp.get_model(input_size=input_size, layer1_size=128, layer2_size=256, output_size=args.latent_dim)
    model = nn.DataParallel(model, device_ids=None)

    if args.use_cuda:
        model = model.cuda()
    return model




