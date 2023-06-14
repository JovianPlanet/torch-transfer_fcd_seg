import os
import datetime

def get_parameters(mode):

    # mode = 'reg' # available modes: 'reg', 'train', 'test'

    model_dims = (128, 128, 64)
    lr         = 0.001
    epochs     = 20
    batch_size = 8
    new_z      = [2, 2, 2]
    n_heads    = 23  #368 Total cabezas disponibles entrenamiento: 295
    n_train    = 19 #295
    n_val      = 2 #37
    n_test     = 2

    labels = {'bgnd': 0, # Image background
              'FCD' : 1, # Focal cortical dysplasia
    }

    model_fn  = 'weights-bcedice-'+str(epochs)+'_eps-'+str(n_train)+'_heads-'+str(datetime.date.today())+'-_bn.pth'
    losses_fn = './outs/losses-bcedice-'+str(epochs)+'_eps-'+str(n_train)+'_heads-'+str(datetime.date.today())+'-_bn.csv'
    dices_fn  = './outs/dices-bcedice-'+str(epochs)+'_eps-'+str(n_train)+'_heads-'+str(datetime.date.today())+'-_bn.csv'

    PATH_PRETRAINED_MODEL = './pretrained/weights-bcedice-20_eps-100_heads-2023-03-10-_nobn.pth'

    PATH_TRAINED_MODEL = 'weights-bcedice-20_eps-100_heads-2023-03-10-_nobn.pth'
    PATH_TEST_DICES = './outs/dice_coeff'+PATH_TRAINED_MODEL[7:-4]+'-test.csv'

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

        return {'mode'        : mode,
                'data'        : datasets,
                'pretrained'  : PATH_PRETRAINED_MODEL,
                'model_dims'  : model_dims,
                'lr'          : lr,
                'epochs'      : epochs,
                'batch_size'  : batch_size,
                'new_z'       : new_z,
                'n_heads'     : n_heads,
                'n_train'     : n_train,
                'n_val'       : n_val,
                'model_fn'    : model_fn,
                'losses_fn'   : losses_fn,
                'dices_fn'    : dices_fn,
                'labels'      : labels,
        }

    elif mode == 'test':

        threshold = 0.5

        return {'mode'      : mode,
                'data'      : datasets,
                'model_dims': model_dims,
                'new_z'     : new_z,
                'n_test'    : n_test,
                'thres'     : threshold,
                'labels'    : labels,
                'weights'   : PATH_TRAINED_MODEL,
                'test_fn'   : PATH_TEST_DICES,
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
