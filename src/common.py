from dataclasses import dataclass, asdict
from itertools import product
from typing import Optional
import copy
import os
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, TensorDataset, Subset
from skimage.metrics import structural_similarity as ssim

from aijack.collaborative.fedavg import FedAVGAPI, FedAVGClient, FedAVGServer
from aijack.attack.inversion import GradientInversionAttackServerManager
from aijack.defense.dp import DPSGDManager, GeneralMomentAccountant
from aijack.defense.dp.manager import DPSGDClientManager

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")