import os
import datetime

def get_parameters(mode):

    # mode = 'reg' # available modes: 'reg', 'train', 'test'

    hyperparams = {'model_dims': (128, 128, 64), # Dimensiones de entrada al modelo
                   'lr'        : 0.0001,         # Taza de aprendizaje
                   'epochs'    : 20,             # Numero de epocas
                   'batch_size': 1,              # Tama;o del batch
                   'new_z'     : [2, 2, 2],      # Nuevo tama;o de zooms
                   'n_heads'   : 23,             # Numero de cabezas
                   'n_train'   : 19,             # "" Entrenamiento
                   'n_val'     : 2,              # "" Validacion
                   'n_test'    : 2,              # "" Prueba
                   'batchnorm' : False           # Normalizacion de batch
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


    if mode == 'train':

        files = {'model': 'weights-BCEDice-Frz_2-'+str(hyperparams['epochs'])+'_eps-'+str(datetime.date.today())+'-_bn', 
                 'losses': './outs/losses-BCEDice-Frz_2-'+str(hyperparams['epochs'])+'_eps-'+str(datetime.date.today())+'-_bn.csv', 
                 't_dices': './outs/t-dices-BCEDice-Frz_2-'+str(hyperparams['epochs'])+'_eps-'+str(datetime.date.today())+'-_bn.csv', 
                 't_accus': './outs/accs-BCEDice-Frz_2-'+str(hyperparams['epochs'])+'_eps-'+str(datetime.date.today())+'-_bn.csv',
                 'v_dices': './outs/v-dices-BCEDice-Frz_2-'+str(hyperparams['epochs'])+'_eps-'+str(datetime.date.today())+'-_bn.csv',
                 'v_accus': './outs/v-accs-BCEDice-Frz_2-'+str(hyperparams['epochs'])+'_eps-'+str(datetime.date.today())+'-_bn.csv',
                 'pics': './outs/imgs/BCEDice-Frz_2-'+str(hyperparams['epochs'])+'_eps-'+str(datetime.date.today())}

        # Modelo preentrenado en segmentacion de tumores
        PATH_PRETRAINED_MODEL = './pretrained/weights-BCEDice-10_eps-25_heads-2023-06-30-_nobn-e9.pth'

        return {'mode'        : mode,
                'data'        : datasets,
                'pretrained'  : PATH_PRETRAINED_MODEL,
                'hyperparams' : hyperparams,
                'files'       : files,
        }

    elif mode == 'test':

        # En este archivo se guarda el modelo de segmentacion de FCD por transfer learning
        PATH_TRAINED_MODEL = 'weights-bce(acc)-20_eps-2023-06-23.pth'

        # CSV con resultados de Dice en test del modelo de segmentacion de FCD
        PATH_TEST_DICES = './outs/dice_coeff'+PATH_TRAINED_MODEL[7:-4]+'-test.csv' # CSV con resultados de Dice en test

        threshold = 0.5

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
