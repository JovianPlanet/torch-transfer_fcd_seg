import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.functional import dice
import pandas as pd
from get_data import Unet2D_DS
from unet import Unet
from metrics import dice_coeff
from utils.plots import plot_batch_full, plot_overlays


def test(config):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing {device} device\n")

    test_ds = Unet2D_DS(config, 'test')

    test_mris = DataLoader(
        test_ds, 
        batch_size=1 #config['batch_size'],
    )

    unet = Unet(num_classes=1, depth=5, batchnorm=config['hyperparams']['batchnorm']).to(device, dtype=torch.double)
    unet.load_state_dict(torch.load(config['weights']))

    print(f'Test del modelo {config["weights"]}\n')

    gen_dice = 0
    dices = []

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        unet.eval()
        for i, data in enumerate(test_mris):
            images, labels = data
            images = images.unsqueeze(1).to(device, dtype=torch.double)
            labels = labels.to(device, dtype=torch.double)

            # calculate outputs by running images through the network
            outputs = unet(images)
            probs = nn.Sigmoid()  # Sigmoid para biclase
            preds  = probs(outputs) 
            preds = torch.where(preds>config['thres'], 1., 0.)

            batch_dice = dice_coeff(preds, labels)
            #batch_dice = dice(preds, labels.long(), ignore_index=0, zero_division=1) # Metrica dice de torchmetrics
            gen_dice += batch_dice.item()
            dices.append([i, batch_dice.item()])
            #print(f'Test Dice score (batch): {batch_dice:.3f}')
            
            if (i + 1) % 100 == 0:
                print(f'Dice promedio despues de {i+1} batches = {gen_dice/(i+1):.3f}')

            #plot_batch_full(images.squeeze(1), labels, preds.squeeze(1))
            #if torch.any(labels):
            print(batch_dice)
            plot_overlays(images.squeeze(1), labels, preds.squeeze(1))
            
    gen_dice = gen_dice / (i+1)

    print(f'\nDice promedio total = {gen_dice:.3f}')
    df_dice = pd.DataFrame(dices, columns=['Batch', 'Dice'])
    df_dice = df_dice.assign(id=df_dice.index.values)
    #df_dice.to_csv(config['test_fn'])

# dices = torch.tensor(dices)
# torch.save(dices, PATH_TEST)