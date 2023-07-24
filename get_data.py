import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import nibabel as nib
import nibabel.processing


class Unet2D_DS(Dataset):

    def __init__(self, config, mode, cropds=False):

        self.config = config
        self.mode   = mode
        self.cropds = cropds

        data_dir = ''
        n = 0

        if self.mode == 'train':
            data_dir = self.config['data']['train'] 
            n = self.config['hyperparams']['n_train']

        elif self.mode == 'val':
            data_dir = self.config['data']['val']
            n = self.config['hyperparams']['n_val']

        elif self.mode == 'test':
            data_dir = self.config['data']['test']
            n = self.config['hyperparams']['n_test']

        self.subjects = next(os.walk(data_dir))[1] # [2]: lists files; [1]: lists subdirectories; [0]: root

        self.L = []

        for subject in self.subjects[:n]:

            mri_path   = os.path.join(data_dir, subject, self.config['data']['mri_fn'])
            label_path = os.path.join(data_dir, subject, self.config['data']['mask_fn'])
            head  = preprocess(label_path, self.config)

            a = True
            for slice_ in range(self.config['hyperparams']['model_dims'][2]):

                if self.cropds:
                
                    if np.any(head[:, :, slice_]):
                        self.L.append([subject, slice_, mri_path, label_path])
                        # if slice_ > 3 and a:
                        #     self.L.append([subject, slice_-1, mri_path, label_path])
                        #     self.L.append([subject, slice_-2, mri_path, label_path])
                        #     self.L.append([subject, slice_-3, mri_path, label_path])
                        #     a = False
                else:
                    self.L.append([subject, slice_, mri_path, label_path])

        self.df = pd.DataFrame(self.L, columns=['Subject', 'Slice', 'Path MRI', 'Path Label'])
        self.df = self.df.assign(id=self.df.index.values).sample(frac=1)
        #print(f'dataframe: \n{self.df} \n')


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
            scan = nib.Nifti1Image(vol, aff)
        except:
            vol = np.where(vol==0., 0., 1.)
            scan = nib.Nifti1Image(vol, aff)


    # Remuestrea volumen y affine a un nuevo shape

    #new_zooms  = np.array(scan.header.get_zooms()) * config['new_z']
    #new_shape  = np.array(vol.shape) // config['new_z']

    new_affine = nibabel.affines.rescale_affine(aff, 
                                                vol.shape, 
                                                config['hyperparams']['new_z'], 
                                                config['hyperparams']['model_dims']
    )

    scan       = nibabel.processing.conform(scan, 
                                            config['hyperparams']['model_dims'], 
                                            config['hyperparams']['new_z']
    )
     
    ni_img     = nib.Nifti1Image(scan.get_fdata(), new_affine)
    vol        = ni_img.get_fdata() 

    if 'Ras_msk' in path:
        vol = np.where(vol <= 0.1, 0., 1.)

    if norm:
        vol = (vol - np.min(vol))/(np.max(vol) - np.min(vol))

    return vol


