from datetime import datetime
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from torchmetrics.classification import BinaryAccuracy, Dice
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

    unet = Unet(1, depth=5, batchnorm=config['hyperparams']['batchnorm']).to(device, dtype=torch.double)
    #print(torch.cuda.memory_summary(device=device, abbreviated=False))

    # Se carga el modelo preentrenado
    unet.load_state_dict(torch.load(config['pretrained']))
    print(f'Modelo preentrenado: {config["pretrained"]}\n')

    summary(unet)#, input_size=(batch_size, 1, 28, 28))

    print(unet)
    print(unet.up_convs)

    # Congelar las capas del modelo
    for param in unet.parameters():
        param.requires_grad = False

    # Anadir nuevas capas entrenables
    unet.conv_final = nn.Conv2d(64, 1, kernel_size=1, groups=1, stride=1).to(device, dtype=torch.double)
    unet.up_convs[3].conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True, groups=1).to(device, dtype=torch.double)
    unet.up_convs[3].conv1 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=True, groups=1).to(device, dtype=torch.double)
    unet.up_convs[3].upconv = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2).to(device, dtype=torch.double)
    summary(unet)

    criterion = {'CELoss' : nn.CrossEntropyLoss(),  # Cross entropy loss performs softmax by default
                 'BCELog' : nn.BCEWithLogitsLoss(), # BCEWithLogitsLoss performs sigmoid by default
                 'BCE'    : nn.BCELoss(),
                 'Dice'   : DiceLoss(),
                 'TmDice' : Dice(threshold=0.1, ignore_index=0),                 # Dice de torchmetrics
                 'BCEDice': BCEDiceLoss(),
                 'Focal'  : FocalLoss(),
                 'Tversky': TverskyLoss()
    }

    optimizer = Adam(unet.parameters(), lr=config['hyperparams']['lr'])
    scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=4)

    #acc = BinaryAccuracy(multidim_average='global').to(device, dtype=torch.double)
    metric = Dice(threshold=0.1, ignore_index=0).to(device, dtype=torch.double)

    best_loss = 1.0
    best_epoch_loss = 0
    best_epoch_dice = 0
    best_dice = None
    best_acc = None

    losses = []
    train_dices = []
    val_dices  = []
    train_accs = []
    val_accs = []

    for epoch in tqdm(range(config['hyperparams']['epochs'])):  # loop over the dataset multiple times

        #torch.cuda.empty_cache()

        running_loss = 0.0
        running_dice = 0.0
        epoch_loss   = 0.0
        ep_tr_dice = 0
        ep_tr_acc = 0

        print(f'\n\nEpoch {epoch + 1}\n')

        unet.train()
        
        for i, data in enumerate(train_dl, 0):

            inputs, labels = data
            inputs = inputs.unsqueeze(1).to(device, dtype=torch.double)
            labels = labels.to(device, dtype=torch.double)

            outputs = unet(inputs)
            #plot_batch(masks_pred, labels)

            loss = criterion['BCEDice'](outputs.double(), labels.unsqueeze(1)) # Utilizar para BCELoss o DiceLoss
            #loss = criterion(outputs, labels.long()) # Utilizar para Cross entropy loss (multiclase)

            '''Metricas'''
            probs_ = nn.Sigmoid()  # Sigmoid para biclase
            pval_  = probs_(outputs) 
            preds_ = torch.where(pval_>0.1, 1., 0.)
            # Dice coefficient
            ba_tr_dice = dice_coeff(preds_, labels.unsqueeze(1))
            ep_tr_dice += ba_tr_dice.item()
            train_dices.append([epoch, i, ba_tr_dice.item()])
            # Accuracy
            ba_tr_acc = torch.sum(preds_ == labels.unsqueeze(1)).item() / (128*128) # acc(preds_, labels.unsqueeze(1)).item()
            ep_tr_acc += ba_tr_acc
            train_accs.append([epoch, i, ba_tr_acc])
            '''Fin metricas'''

            running_loss += loss.item()
            optimizer.zero_grad()            # zero the parameter gradients
            loss.backward(retain_graph=True) # retain_graph=True
            optimizer.step()

            losses.append([epoch, i, loss.item()])

            if (i+1) % 50 == 0: 
                print(f'Batch No. {i+1}')
                print(f'Loss = {running_loss/(i+1):.3f}')
                print(f'Accuracy (prom) = {ep_tr_acc/(i+1):.3f}, Dice (prom) = {ep_tr_dice/(i+1):.3f}')

        before_lr = optimizer.param_groups[0]["lr"]
        if (epoch + 1) % 5 == 0: 
            scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
            
        epoch_loss = running_loss/(i + 1)  

        epoch_val_dice = 0 
        epoch_val_acc = 0  

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
                preds = torch.where(pval>0.1, 1., 0.)
                #preds = torch.argmax(pval, dim=1)

                # Dice coefficient
                batch_val_dice = dice_coeff(preds, y.unsqueeze(1))
                #batch_val_dice = dice(preds.squeeze(1), y.long(), ignore_index=0, zero_division=1) # version de torchmetrics de la metrica
                epoch_val_dice += batch_val_dice.item()
                val_dices.append([epoch, j, batch_val_dice.item()])
                # Dice torchmetrics
                metric.update(preds, y.unsqueeze(1).long())

                # Accuracy
                batch_val_acc = torch.sum(preds == y.unsqueeze(1)).item() / (128*128) #acc(preds.squeeze(1), y).item()
                epoch_val_acc += batch_val_acc
                val_accs.append([epoch, j, batch_val_acc])

                if (j+1) % 20 == 0: 
                    print(f'pval min = {pval.min():.3f}, pval max = {pval.max():.3f}')
                    print(f'Dice promedio hasta batch No. {j+1} = {epoch_val_dice/(j+1):.3f}')
                    print(f'Dice torchmetrics hasta batch No. {j+1} = {metric.compute():.3f}')
                    print(f'Accuracy promedio hasta batch No. {j+1} = {epoch_val_acc/(j+1):.3f}')

                #if (j+1) % 8 == 0:
                if torch.any(y):
                    plot_overlays(x.squeeze(1), 
                                  y, 
                                  preds.squeeze(1), 
                                  mode='save', 
                                  fn=f"{config['files']['pics']}-epoca_{epoch + 1}-b{j}.pdf")

        epoch_val_dice = epoch_val_dice / (j + 1) 
        epoch_val_acc  = epoch_val_acc / (j + 1)

        if epoch == 0:
            best_loss = epoch_loss
            best_epoch_loss = epoch + 1
            best_dice = epoch_val_dice
            best_epoch_dice = epoch + 1
            best_acc = epoch_val_acc

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch_loss = epoch + 1

        print(f'\nEpoch loss = {epoch_loss:.3f}, Best loss = {best_loss:.3f} (epoca {best_epoch_loss})')
        print(f'Epoch dice (Training) = {ep_tr_dice / (i+1):.3f}')
        print(f'Epoch accuracy (Training) = {ep_tr_acc / (i+1):.3f}\n')

        if epoch_val_dice > best_dice:
            best_dice = epoch_val_dice
            best_epoch_dice = epoch + 1
            #print(f'\nUpdated weights file!')
        torch.save(unet.state_dict(), config['files']['model']+f'-e{epoch+1}.pth')

        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc

        print(f'\nEpoch dice (Validation) = {epoch_val_dice:.3f}, Best dice = {best_dice:.3f} (epoca {best_epoch_dice})')
        print(f'Epoch Dice torchmetrics (Validation) = {metric.compute():.3f}')
        print(f'Epoch accuracy (Validation) = {epoch_val_acc:.3f}, Best accuracy = {best_acc:.3f}')
        print(f'lr = {before_lr} -> {after_lr}\n')
        metric.reset()

    df_loss = pd.DataFrame(losses, columns=['Epoca', 'Batch', 'Loss'])
    df_loss = df_loss.assign(id=df_loss.index.values)
    df_loss.to_csv(config['files']['losses'])

    df_train_dice = pd.DataFrame(val_dices, columns=['Epoca', 'Batch', 'Dice'])
    df_train_dice = df_train_dice.assign(id=df_train_dice.index.values)
    df_train_dice.to_csv(config['files']['t_dices'])

    df_train_acc = pd.DataFrame(val_accs, columns=['Epoca', 'Batch', 'Accuracy'])
    df_train_acc = df_train_acc.assign(id=df_train_acc.index.values)
    df_train_acc.to_csv(config['files']['t_accus'])

    df_val_dice = pd.DataFrame(val_dices, columns=['Epoca', 'Batch', 'Dice'])
    df_val_dice = df_val_dice.assign(id=df_val_dice.index.values)
    df_val_dice.to_csv(config['files']['v_dices'])

    df_val_acc = pd.DataFrame(val_accs, columns=['Epoca', 'Batch', 'Accuracy'])
    df_val_acc = df_val_acc.assign(id=df_val_acc.index.values)
    df_val_acc.to_csv(config['files']['v_accus'])

    print(f'\nFinished training. Total training time: {datetime.now() - start_time}\n')


#nn.utils.clip_grad_norm_(unet.parameters(), max_norm=2.0, norm_type=2)
#nn.utils.clip_grad_value_(unet.parameters(), clip_value=1.0) # Gradient clipping