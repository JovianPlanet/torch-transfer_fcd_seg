from datetime import datetime
#import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torchmetrics.functional import dice
from torchinfo import summary
from unet import Unet
from metrics import DiceLoss, dice_coeff, BCEDiceLoss, FocalLoss, TverskyLoss
from get_data import Unet2D_DS



def train(config):

    torch.cuda.empty_cache()

    print(f'\nHora de inicio: {datetime.now()}')

    print(f"\nNum epochs = {config['epochs']}, batch size = {config['batch_size']}, Learning rate = {config['lr']}, \
        file name={config['model_fn']}\n")

    # Datasets #
    ds_train = Unet2D_DS(config, 'train')
    ds_val   = Unet2D_DS(config, 'val')

    train_dl = DataLoader(
        ds_train, #train_mris, 
        batch_size=config['batch_size'],
        shuffle=True,
    )

    val_dl = DataLoader(
        ds_val, #val_mris, 
        batch_size=1 #config['batch_size'],
    )

    print(f'Tamano del dataset de entrenamiento (IATM): {len(ds_train)} slices')
    print(f'Tamano del dataset de validacion: {len(ds_val)} slices \n')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    unet = Unet(1, depth=5).to(device, dtype=torch.double)
    #print(torch.cuda.memory_summary(device=device, abbreviated=False))

    # Se carga el modelo preentrenado
    unet.load_state_dict(torch.load(config['pretrained']))
    print(f'Modelo preentrenado: {config["pretrained"]}\n')

    summary(unet)#, input_size=(batch_size, 1, 28, 28))

    print(unet.conv_final)


    for param in unet.parameters():
        param.requires_grad = False
    unet.fc = nn.Conv2d(64, 10, kernel_size=1)
    summary(unet)

    criterion = {'CELoss' : nn.CrossEntropyLoss(),  # Cross entropy loss performs softmax by default
                 'BCELog' : nn.BCEWithLogitsLoss(), # BCEWithLogitsLoss performs sigmoid by default
                 'BCE'    : nn.BCELoss(),
                 'Dice'   : DiceLoss(),
                 'BCEDice': BCEDiceLoss(),
                 'Focal'  : FocalLoss(),
                 'Tversky': TverskyLoss()
    }

    optimizer = Adam(unet.parameters(), lr=config['lr'])

    best_loss = 1.0

    losses = []
    dices  = []