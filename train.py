from datetime import datetime
#import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torchmetrics.functional import dice
from unet import Unet
from metrics import DiceLoss, dice_coeff, BCEDiceLoss, FocalLoss, TverskyLoss
from get_data import Unet2D_DS



def train(config):

    torch.cuda.empty_cache()

    print(f'\nHora de inicio: {datetime.now()}')

    print(f"\nNum epochs = {config['epochs']}, batch size = {config['batch_size']}, Learning rate = {config['lr']}, \
        file name={config['model_fn']}\n")

    # Crear datasets #

    ds_train = Unet2D_DS(config, 'train')
    ds_val   = Unet2D_DS(config, 'val')

    # train_size = int(0.8 * len(ds))
    # test_size  = len(ds) - train_size

    # train_mris, val_mris = random_split(ds, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    train_dl = DataLoader(
        ds_train, #train_mris, 
        batch_size=config['batch_size'],
        shuffle=True,
    )

    val_dl = DataLoader(
        ds_val, #val_mris, 
        batch_size=1 #config['batch_size'],
    )