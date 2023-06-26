from datetime import datetime
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.functional import dice
from torchinfo import summary
from unet import Unet
from metrics import DiceLoss, dice_coeff, BCEDiceLoss, FocalLoss, TverskyLoss
from get_data import Unet2D_DS
from utils.plots import plot_overlays


def train(config):

    torch.cuda.empty_cache()

    start_time = datetime.now()

    print(f'\nHora de inicio: {start_time}')

    print(f"\nEpocas = {config['hyperparams']['epochs']}, batch size = {config['hyperparams']['batch_size']}, \
            Learning rate = {config['hyperparams']['lr']}\n")

    print(f"Nombre de archivo del modelo: {config['files']['model']}\n")

    # Datasets #
    ds_train = Unet2D_DS(config, 'train')
    ds_val   = Unet2D_DS(config, 'val')

    train_dl = DataLoader(
        ds_train, #train_mris, 
        batch_size=config['hyperparams']['batch_size'],
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
    print(unet.up_convs)

    # Congelar las capas del modelo
    for param in unet.parameters():
        param.requires_grad = False

    # Poner una nueva capa entrenable al final
    unet.conv_final = nn.Conv2d(64, 1, kernel_size=1, groups=1, stride=1).to(device, dtype=torch.double)
    unet.up_convs[3] = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
    summary(unet)

    criterion = {'CELoss' : nn.CrossEntropyLoss(),  # Cross entropy loss performs softmax by default
                 'BCELog' : nn.BCEWithLogitsLoss(), # BCEWithLogitsLoss performs sigmoid by default
                 'BCE'    : nn.BCELoss(),
                 'Dice'   : DiceLoss(),
                 'BCEDice': BCEDiceLoss(),
                 'Focal'  : FocalLoss(),
                 'Tversky': TverskyLoss()
    }

    optimizer = Adam(unet.parameters(), lr=config['hyperparams']['lr'])

    acc = BinaryAccuracy(multidim_average='global').to(device, dtype=torch.double)

    best_loss = 1.0
    best_dice = None
    best_acc = None

    losses = []
    dices  = []
    accuracies = []

    for epoch in tqdm(range(config['hyperparams']['epochs'])):  # loop over the dataset multiple times

        #torch.cuda.empty_cache()

        running_loss = 0.0
        running_dice = 0.0
        epoch_loss   = 0.0
        
        print(f'\n\nEpoch {epoch + 1}\n')

        unet.train()
        
        for i, data in enumerate(train_dl, 0):

            inputs, labels = data
            inputs = inputs.unsqueeze(1).to(device, dtype=torch.double)
            labels = labels.to(device, dtype=torch.double)

            outputs = unet(inputs)
            #plot_batch(masks_pred, labels)

            loss = criterion['BCEDice'](outputs.double().squeeze(1), labels) # Utilizar esta linea para BCELoss o DiceLoss
            #loss = criterion(outputs, labels.long()) # Utilizar esta linea para Cross entropy loss (multiclase)

            if (i+1) % 50 == 0: 
                print(f'Batch No. {(i+1)} loss = {loss.item():.3f}')

            running_loss += loss.item()
            optimizer.zero_grad()            # zero the parameter gradients
            loss.backward(retain_graph=True) # retain_graph=True
            
            optimizer.step()
            losses.append([epoch, i, loss.item()])
            
        epoch_loss = running_loss/(i + 1)  

        epoch_dice = 0 
        epoch_acc = 0  
        pvalmax = []

        with torch.no_grad():
            unet.eval()
            print(f'\nValidacion\n')
            for j, valdata in enumerate(val_dl):
                x, y = valdata
                x = x.unsqueeze(1).to(device, dtype=torch.double)
                y = y.to(device, dtype=torch.double)

                outs  = unet(x)
                #probs = nn.Softmax(dim=1) # Softmax para multiclase
                probs = nn.Sigmoid()  # Sigmoid para biclase
                pval  = probs(outs) 
                preds = torch.where(pval>0.5, 1., 0.)
                #preds = torch.argmax(pval, dim=1)

                # Dice coefficient
                batch_dice = dice_coeff(preds.squeeze(1), y)
                #batch_dice = dice(preds.squeeze(1), y.long(), ignore_index=0, zero_division=1) # version de torchmetrics de la metrica
                epoch_dice += batch_dice.item()
                dices.append([epoch, j, batch_dice.item()])

                # Accuracy
                batch_acc = acc(preds.squeeze(1), y).item()
                epoch_acc += batch_acc
                accuracies.append([epoch, j, batch_acc])

                if (j+1) % 20 == 0: 
                    print(f'pval min = {pval.min():.3f}, pval max = {pval.max():.3f}')
                    print(f'Dice promedio hasta batch No. {j+1} = {epoch_dice/(j+1):.3f}')
                    print(f'Accuracy promedio hasta batch No. {j+1} = {epoch_acc/(j+1):.3f}')

                if torch.any(y):
                    plot_overlays(x.squeeze(1), 
                                  y, 
                                  preds.squeeze(1), 
                                  mode='save', 
                                  fn=f"{config['files']['pics']}-epoca_{epoch + 1}-b{j}")

        epoch_dice = epoch_dice / (j + 1) 
        epoch_acc  = epoch_acc / (j + 1)

        if epoch == 0:
            best_loss = epoch_loss
            best_dice = epoch_dice
            best_acc = epoch_acc

        if epoch_loss < best_loss:
            best_loss = epoch_loss

        print(f'\nEpoch loss = {epoch_loss:.3f}, Best loss = {best_loss:.3f}\n')

        if epoch_dice > best_dice:
            best_dice = epoch_dice
            print(f'\nUpdated weights file!')
            torch.save(unet.state_dict(), config['files']['model'])

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            # print(f'\nUpdated weights file!')
            # torch.save(unet.state_dict(), config['files']['model'])

        print(f'\nEpoch dice (Validation) = {epoch_dice:.3f}, Best dice = {best_dice:.3f}')
        print(f'Epoch accuracy (Validation) = {epoch_acc:.3f}, Best dice = {best_acc:.3f}\n')
        print(f'Elapsed time: {datetime.now() - start_time}\n')

    df_loss = pd.DataFrame(losses, columns=['Epoca', 'Batch', 'Loss'])
    df_loss = df_loss.assign(id=df_loss.index.values)
    df_loss.to_csv(config['files']['losses'])

    df_dice = pd.DataFrame(dices, columns=['Epoca', 'Batch', 'Dice'])
    df_dice = df_dice.assign(id=df_dice.index.values)
    df_dice.to_csv(config['files']['dices'])

    df_acc = pd.DataFrame(accuracies, columns=['Epoca', 'Batch', 'Accuracy'])
    df_acc = df_acc.assign(id=df_acc.index.values)
    df_acc.to_csv(config['files']['accus'])

    print(f'\nDuration: {datetime.now() - start_time}')


#nn.utils.clip_grad_norm_(unet.parameters(), max_norm=2.0, norm_type=2)
#nn.utils.clip_grad_value_(unet.parameters(), clip_value=1.0) # Gradient clipping