# import openslide 
import glob
import numpy as np
# import h5py
# import PIL
import pandas
# import cv2
# import os 
import torch
import random
import torchvision.transforms as T
import PIL

from PIL import Image
Image.MAX_IMAGE_PIXELS = None
# from feature_selection import top_gene_selection

class HBCDataset(torch.utils.data.Dataset):
    def __init__(self, path='/newdata/why/Human_breast_cancer_in_situ_capturing_transcriptomics/BRCA',mode='train',selected_genes=None,patch_size=224):
        # selected_genes=top_gene_selection(path,2000)
        if selected_genes==None:
            selected_genes=np.load('HBC_Selected_Genes.npy',allow_pickle=True).tolist()

        self.verbose=False
        self.mode=mode
        self.wsi_imgs=[]
        self.wsi_imgs2=[]
        self.wsi_imgs3=[]
        self.st_coords=[]
        self.st_feats=[]
        self.st_spots_pixel_map=[]
        # self.transform=T.Compose([
        #     T.RandomHorizontalFlip(0.5),
        #     T.RandomVerticalFlip(0.5),
        #     T.RandomRotation(degrees=(0, 360),fill=255),
        #     T.ToTensor()]
        #     )
        self.transform=T.Compose([
            T.ToTensor()
        ])

        wsi_filenames=sorted(glob.glob(path+'/*/*.jpg'))
        random.seed(1553)
        random.shuffle(wsi_filenames)
        random.seed()

        train_frac, val_frac, test_frac = 0.6, 0.1, 0.3
        n_train=int(len(wsi_filenames)*train_frac) 
        n_val = int(len(wsi_filenames)*val_frac)
        n_test=int(len(wsi_filenames)*test_frac)

        if mode=='train':
            wsi_filenames=wsi_filenames[:n_train]
        elif mode=='val':
            wsi_filenames=wsi_filenames[n_train:n_train+n_val]
        elif mode=='test':
            wsi_filenames=wsi_filenames[n_train+n_val:]

        for wsi_file in wsi_filenames:
            print('processing:', wsi_file)
            wsi_img=PIL.Image.open(wsi_file)

            wsi_basename=wsi_file.split(r'.')[0]
            st_coords=wsi_basename+'_Coord.tsv' # stores label, useless in SR
            st_feats=wsi_basename+'.tsv'
            st_spots_pixel_map=wsi_basename+'.spots.txt'

            feats_all=pandas.read_csv(st_feats,sep='\t',index_col=0,header=0)
            spots_pixel_map_all=pandas.read_csv(st_spots_pixel_map,sep=',',index_col=0,header=0)

            for spots_pixel_map in spots_pixel_map_all.iterrows():
                idx=spots_pixel_map[0]
                x=spots_pixel_map[1]['X']
                y=spots_pixel_map[1]['Y']

                try:
                    patch=wsi_img.crop((int(x-patch_size//2), int(y-patch_size//2),int(x+patch_size//2),int(y+patch_size//2)))
                    if patch.size[0]!=patch_size or patch.size[1]!=patch_size:
                        print(idx,':',patch.size)
                        raise Exception
                except Exception as e:
                    if self.verbose:
                        print('patch shape error encountered:',e, idx)
                    continue

                try:
                    feats=np.array(feats_all.loc[idx][selected_genes])
                except Exception as e:
                    if self.verbose:
                        print('feats error encountered:',e, idx)
                    continue
                spot_sum=np.sum(feats)
                if spot_sum==0:
                    if self.verbose:
                        print('spot_sum=0. skipped:',idx)
                    continue
                feats=np.log1p(feats*1000000/spot_sum)
                self.st_spots_pixel_map.append([idx,x,y])
                self.st_feats.append(np.array(feats))
                self.wsi_imgs.append(patch)
                self.wsi_imgs2.append(patch.resize((patch_size//2,patch_size//2),Image.BILINEAR))
                self.wsi_imgs3.append(patch.resize((patch_size//4,patch_size//4),Image.BILINEAR))

        print('total number of samples:', len(self.st_spots_pixel_map))
    def __len__(self):
        return len(self.st_feats)

    def augmentation(self,img, img2, img3):
        if random.random()<0.5:
            img=img.transpose(Image.FLIP_LEFT_RIGHT)
            img2=img2.transpose(Image.FLIP_LEFT_RIGHT)
            img3=img3.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random()<0.5:
            img=img.transpose(Image.FLIP_TOP_BOTTOM)
            img2=img2.transpose(Image.FLIP_TOP_BOTTOM)
            img3=img3.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random()<0.5:
            degree=random.randint(0,360)
            img=T.RandomRotation((degree,degree),fill=255)(img)
            img2=T.RandomRotation((degree,degree),fill=255)(img2)
            img3=T.RandomRotation((degree,degree),fill=255)(img3)
        return img,img2,img3
    
    def __getitem__(self,index):
        if self.mode=='test':
            return self.st_spots_pixel_map[index], self.transform(self.wsi_imgs[index]),self.transform(self.wsi_imgs2[index]),self.transform(self.wsi_imgs3[index]), self.st_feats[index]
        # patch_info=self.wsi_imgs[index]
        # x,y=patch_info['coord']
        # patch=cv2.imread(patch_info['img'])[int(x-112):int(x+112), int(y-112):int(y+112),:]
        img,img2,img3=self.augmentation(self.wsi_imgs[index],self.wsi_imgs2[index],self.wsi_imgs3[index])
        return self.st_spots_pixel_map[index], self.transform(img), self.transform(img2),self.transform(img3),self.st_feats[index]

        # return self.st_spots_pixel_map[index], self.wsi_imgs[index].transpose(2,0,1), self.st_feats[index]

if __name__=="__main__":
    testdataset=HBCDataset()
