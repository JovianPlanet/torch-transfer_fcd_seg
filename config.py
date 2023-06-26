import os
import datetime

def get_parameters(mode):

    # mode = 'reg' # available modes: 'reg', 'train', 'test'

    hyperparams = {'model_dims': (128, 128, 64),
                   'lr'        : 0.0001,
                   'epochs'    : 20, #20
                   'batch_size': 8,
                   'new_z'     : [2, 2, 2],
                   'n_heads'   : 23,
                   'n_train'   : 19, #19,
                   'n_val'     : 2,
                   'n_test'    : 2
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

        files = {'model' : 'weights-bce(acc)-'+str(hyperparams['epochs'])+'_eps-'+str(datetime.date.today())+'.pth', 
                 'losses': './outs/losses-bce(acc)-'+str(hyperparams['epochs'])+'_eps-'+str(datetime.date.today())+'.csv', 
                 'dices' : './outs/dices-bce(acc)-'+str(hyperparams['epochs'])+'_eps-'+str(datetime.date.today())+'.csv', 
                 'accus' : './outs/accs-bce(acc)-'+str(hyperparams['epochs'])+'_eps-'+str(datetime.date.today())+'.csv',
                 'pics'  : './outs/imgs/bce(acc)-'+str(hyperparams['epochs'])+'_eps-'+str(datetime.date.today())}

        # Modelo preentrenado en segmentacion de tumores
        PATH_PRETRAINED_MODEL = './pretrained/weights-bcedice-20_eps-100_heads-2023-03-10-_nobn.pth'

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
