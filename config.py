import os
from datetime import datetime
from pathlib import Path


def get_parameters(mode):

    # mode = 'reg' # available modes: 'reg', 'train', 'test'

    hyperparams = {'model_dims': (128, 128, 64), # Dimensiones de entrada al modelo
                   'new_z'     : [2, 2, 2],      # Nuevo tama;o de zooms
                   'lr'        : 0.0005,         # Taza de aprendizaje
                   'epochs'    : 20,             # Numero de epocas
                   'batch_size': 1,              # Tama;o del batch
                   'crit'      : 'BCELog',       # Fn de costo. Opciones: 'CELoss', 'BCELog', 'BCELogW', 'BCEDice', 'Dice'
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
    img_folder = os.path.join(folder, 'imgs')

    Path(folder).mkdir(parents=True, exist_ok=True)
    Path(img_folder).mkdir(parents=True, exist_ok=True)

    if mode == 'train':

        files = {'model'  : os.path.join(folder, 'weights'),
                 'losses' : os.path.join(folder, 'losses.csv'),
                 't_dices': os.path.join(folder, 't_dices.csv'),
                 't_accus': os.path.join(folder, 'train_acc.csv'),
                 'v_dices': os.path.join(folder, 'v_dices.csv'),
                 'v_accus': os.path.join(folder, 'val_acc.csv'),
                 'pics'   : os.path.join(img_folder, 'seg'),
                 'params' : os.path.join(folder, 'params.txt'),
                 'summary': os.path.join(folder, 'cnn_summary.txt')}

        # files = {'model': 'weights-BCEDice-Frz_4-w5-''-_nobn', 
        #          'losses': './outs/losses-BCEDice-Frz_4-w5-''-_nobn.csv', 
        #          't_dices': './outs/t-dices-BCEDice-Frz_4-w5-''-_nobn.csv', 
        #          't_accus': './outs/t-accs-BCEDice-Frz_4-w5-''-_nobn.csv',
        #          'v_dices': './outs/v-dices-BCEDice-Frz_4-w5-''-_nobn.csv',
        #          'v_accus': './outs/v-accs-BCEDice-Frz_4-w5-''-_nobn.csv',
        #          'pics': './outs/imgs/BCEDice-Frz_4-w5-'+str(hyperparams['epochs'])+'_eps-'+str(datetime.date.today())}

        # Modelo preentrenado en segmentacion de tumores
        PATH_PRETRAINED_MODEL = './pretrained/weights-BCEDice-20_eps-100_heads-2023-07-03-_nobn-e20.pth'

        return {'mode'        : mode,
                'data'        : datasets,
                'pretrained'  : PATH_PRETRAINED_MODEL,
                'hyperparams' : hyperparams,
                'files'       : files,
        }

    elif mode == 'test':

        # En este archivo se guarda el modelo de segmentacion de FCD por transfer learning
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
