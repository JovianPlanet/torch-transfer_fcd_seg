import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import nibabel as nib
import nibabel.processing


class Unet2D_DS(Dataset):

    def __init__(self, config, mode):

        self.config = config
        self.mode   = mode

        data_dir = ''
        n = 0

        if self.mode == 'train':
            data_dir = self.config['data']['train'] 
            n = self.config['n_train']

        elif self.mode == 'val':
            data_dir = self.config['data']['val']
            n = self.config['n_val']

        elif self.mode == 'test':
            data_dir = self.config['data']['test']
            n = self.config['n_test']

        self.subjects = next(os.walk(data_dir))[1] # [2]: lists files; [1]: lists subdirectories; [0]: root

        self.L = []

        for subject in self.subjects[:n]:
            files = next(os.walk(os.path.join(data_dir, subject)))[2]

            mri_path   = os.path.join(data_dir, subject, self.config['data']['mri_fn'])
            label_path = os.path.join(data_dir, subject, self.config['data']['mask_fn'])
            vol  = nib.load(label_path).get_fdata()
            for slice_ in range(vol.shape[2]): #(self.config['model_dims'][2]):
                
                if np.any(vol[:,:,slice_]):
                    self.L.append([subject, slice_, mri_path, label_path])

        self.df = pd.DataFrame(self.L, columns=['Subject', 'Slice', 'Path MRI', 'Path Label'])
        self.df = self.df.assign(id=self.df.index.values).sample(frac=1)
        print(f'dataframe: \n{self.df} \n')


    def __len__(self):

        return self.df.shape[0]


    def __getitem__(self, index):

        load_slice = self.df.at[index, 'Slice']

        mri    = preprocess(self.df.at[index, 'Path MRI'], self.config, norm=True)[:, :, load_slice]
        label  = preprocess(self.df.at[index, 'Path Label'], self.config)[:, :, load_slice]

        return mri, label



def preprocess(path, config, norm=False):

    scan = nib.load(path)
    aff  = scan.affine

    vol  = scan.get_fdata()

    if 'Ras_msk' in path:

        try:
            vol = scan.get_fdata().squeeze(3)
            print(f'1 ={np.unique(vol)}')
            scan = nib.Nifti1Image(vol, aff)
            # print(f'2 = {np.unique(scan.get_fdata())}')
        except:
            print(f'3 ={np.unique(vol)}')
            vol = np.where(vol==0., 0., 1.)
            scan = nib.Nifti1Image(vol, aff)
            print(f'4 ={np.unique(scan.get_fdata())}')

    #vol  = scan.get_fdata()#np.int16(scan.get_fdata())

    # Remuestrea volumen y affine a un nuevo shape
    #new_zooms  = np.array(scan.header.get_zooms()) * config['new_z']
    #new_shape  = np.array(vol.shape) // config['new_z']
    new_affine = nibabel.affines.rescale_affine(aff, vol.shape, config['new_z'], config['model_dims']) #new_zooms, (128, 128, 64))#new_shape)
    scan       = nibabel.processing.conform(scan, config['model_dims'], config['new_z']) #(128, 128, 64), new_zooms)
    ni_img     = nib.Nifti1Image(scan.get_fdata(), new_affine)
    vol        = ni_img.get_fdata() #np.int16(ni_img.get_fdata())
    if 'Ras_msk' in path:
        vol = np.where(vol<=0.1, 0., 1.)
        print(f'5 ={np.unique(vol)}')
    if norm:
        vol        = (vol - np.min(vol))/(np.max(vol) - np.min(vol))

    return vol

# hay algunas etiquetas que no estan en el rango [0, 1] sino en el rango [0 255], binarizarlas

'''
Junio 13 2023: Muestra error ya que se enumeran los slices de acuerodo con las dimensiones de la imagen original,
que no necesariamente son iguales a la imagen remuestreada (128x128x64)
'''