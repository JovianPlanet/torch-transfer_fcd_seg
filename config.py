import os
from datetime import datetime
from pathlib import Path


def get_parameters(mode):

    # mode = 'reg' # available modes: 'reg', 'train', 'test'

    hyperparams = {'model_dims': (128, 128, 64), # Dimensiones de entrada al modelo
                   'new_z'     : [2, 2, 2],      # Nuevo tama;o de zooms
                   'lr'        : 0.0005,#0.0005,         # Taza de aprendizaje
                   'epochs'    : 20,             # Numero de epocas
                   'batch_size': 1,              # Tama;o del batch
                   'crit'      : 'BCEDice',     # Fn de costo. Opciones: 'BCEDiceW','BCELog','BCELogW','BCEDice','Dice'
                   'n_train'   : 19,             # "" Entrenamiento
                   'n_val'     : 2,              # "" Validacion
                   'n_test'    : 2,              # "" Prueba
                   'batchnorm' : False,          # Normalizacion de batch
                   'nclasses'  : 1,              # Numero de clases
                   'thres'     : 0.5,            # Umbral
                   'class_w'   : 5.,             # Peso ponderado de la clase
                   'crop'      : True,           # Recortar o no recortar slices sin fcd del volumen
                   'capas'     : 4               # Numero de capas entrenables
    }

    labels = {'bgnd': 0, # Image background
              'FCD' : 1, # Focal cortical dysplasia
    }

    iatm_train = os.path.join('./data',
                               'train',
    )

    iatm_val = os.path.join('./data',
                             'val',
    )

    iatm_test = os.path.join('./data',
                             'test',
    )

    mri_fn  = 'Ras_t1.nii.gz'
    mask_fn = 'Ras_msk.nii.gz'

    datasets = {'train': iatm_train, 'val': iatm_val, 'test': iatm_test, 'mri_fn': mri_fn, 'mask_fn': mask_fn}

    folder = './outs/Ex-' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    if mode == 'train':

        Path(folder).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(folder, 'val_imgs')).mkdir(parents=True, exist_ok=True)

        files = {'model'  : os.path.join(folder, 'weights'),
                 'losses' : os.path.join(folder, 'losses.csv'),
                 't_mets' : os.path.join(folder, 'train_metrics.csv'),
                 'v_mets' : os.path.join(folder, 'val_metrics.csv'),
                 'pics'   : os.path.join(folder, 'val_imgs', 'img'),
                 'params' : os.path.join(folder, 'params.txt'),
                 'summary': os.path.join(folder, 'cnn_summary.txt'),
                 'log'    : os.path.join(folder, 'train.log')
                }

        # Modelo preentrenado en segmentacion de tumores
        #PATH_PRETRAINED_MODEL = './pretrained/Ex-2023-07-18-13-56-48weights-e11.pth' 
        PATH_PRETRAINED_MODEL = './pretrained/weights-BCEDice-20_eps-100_heads-2023-07-03-_nobn-e20.pth'

        return {'mode'        : mode,
                'data'        : datasets,
                'pretrained'  : PATH_PRETRAINED_MODEL,
                'hyperparams' : hyperparams,
                'files'       : files,
        }

    elif mode == 'test':

        PATH_TRAINED_MODEL = './outs/imgs/weights-BCELog-Frz_4-w5-20_eps-2023-07-09-_nobnFCDe6(muy bueno, no se si el mejor)/weights-BCELog-Frz_4-20_eps-2023-07-07-_nobn-e20.pth'

        # CSV con resultados de Dice en test del modelo de segmentacion de FCD
        PATH_TEST_DICES = './outs/dice_coeff'+PATH_TRAINED_MODEL[7:-4]+'-test.csv' # CSV con resultados de Dice en test

        threshold = 0.1

        return {'mode'       : mode,
                'data'       : datasets,
                'hyperparams': hyperparams,
                'thres'      : threshold,
                'labels'     : labels,
                'weights'    : PATH_TRAINED_MODEL,
                'test_fn'    : PATH_TEST_DICES,
        }

    elif mode == 'assess':

        train_losses = './outs/losses-bcedice-20_eps-100_heads-2023-03-10-_nobn.csv'
        train_dices  = './outs/dices-bcedice-20_eps-100_heads-2023-03-10-_nobn.csv'
        test_dices = './outs/dice_coeff-bcedice-20_eps-100_heads-2023-03-10-_nobn-test.csv'

        files = {'train_Loss': train_losses,
                 'train_Dice': train_dices,
                 'test_Dice' : test_dices}

        return {'mode'     : mode,
                'labels'   : labels,
                'losses_fn': losses_fn,
                'dices_fn' : dices_fn,
                'files'    : files}
