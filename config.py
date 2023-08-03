import os
from datetime import datetime
from pathlib import Path


def get_parameters(mode):

    # mode = 'reg' # available modes: 'reg', 'train', 'test'

    hyperparams = {'model_dims': (128, 128, 64), # Dimensiones de entrada al modelo
                   'new_z'     : [2, 2, 2],      # Nuevo tama;o de zooms
                   'lr'        : 0.001,          # Taza de aprendizaje
                   'epochs'    : 50,             # Numero de epocas
                   'batch_size': 1,              # Tama;o del batch
                   'crit'      : 'Focal',        # Fn de costo. Opciones: 'BCEDiceW','BCELog','BCELogW','BCEDice','Dice', 'Focal'
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

        ex = './outs/Ex-2023-07-31-23-29-24'#Ex-2023-07-29-14-57-27'#Ex-2023-07-27-19-34-39' Replica del primero mejor
        # Ex-2023-07-27-19-34-39
        # Ex-2023-07-29-09-14-58
        # Ex-2023-07-29-14-57-27
        # Ex-2023-07-31-00-13-12
        # Ex-2023-07-31-23-29-24
        mo = 'weights-e50.pth'

        test_folder = os.path.join(ex, 'test'+mo[:-4])
        Path(test_folder).mkdir(parents=True, exist_ok=True)
        img_folder = os.path.join(test_folder, 'imgs')
        Path(img_folder).mkdir(parents=True, exist_ok=True)

        PATH_TRAINED_MODEL = os.path.join(ex, mo) #'./outs/Ex/prueba.pth' # 'weights-bcedice-20_eps-100_heads-2023-03-10-_nobn.pth'
        PATH_TEST_METS = os.path.join(test_folder, mo+'-test_metrics.csv')#'./outs/Ex/test_metrics.csv'

        return {'mode'       : mode,
                'data'       : datasets,
                'hyperparams': hyperparams,
                'labels'     : labels,
                'weights'    : PATH_TRAINED_MODEL,
                'test_fn'    : PATH_TEST_METS,
                'img_folder' : img_folder,
        }

    elif mode == 'assess':

        ex = './outs/Ex-2023-07-31-23-29-24'#Ex-2023-07-29-14-57-27'#Ex-2023-07-27-19-34-39'
        ep = 50
        mo = f'weights-e{ep}.pth'

        plots_folder = os.path.join(ex, 'plots-weights_'+mo[-7:-4])

        Path(plots_folder).mkdir(parents=True, exist_ok=True)

        train_losses = 'losses.csv'
        train_mets  = 'train_metrics.csv'
        val_mets = 'val_metrics.csv'
        test_mets = f'weights-e{ep}.pth-test_metrics.csv'

        files = {'train_Loss': os.path.join(ex, train_losses),
                 'train_mets': os.path.join(ex, train_mets),
                 'val_mets'  : os.path.join(ex, val_mets),
                 'test_mets' : os.path.join(ex, f'testweights-e{ep}', test_mets)
        }

        return {'mode'     : mode,
                'labels'   : labels,
                'files'    : files,
                'plots'    : plots_folder,
                'hyperparams': hyperparams,
        }
