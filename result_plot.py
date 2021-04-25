import torch
import torch.backends.cudnn as cudnn

import os
import numpy as np
import argparse

from test import get_normal_vector, split_acc_diff_threshold, cal_score
from utils.utils import adjust_learning_rate, AverageMeter, Logger, get_fusion_label, l2_normalize,\
    post_process, evaluate, get_score
from nce_average import NCEAverage
from nce_criteria import NCECriterion
from utils.setup_NSL import NSL_KDD, NSL_data
from model import generate_model
from models import mlp
# from models import resnet, shufflenet, shufflenetv2, mobilenet, mobilenetv2
import ast

