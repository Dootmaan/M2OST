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
import os
import PIL
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import torchvision.transforms as T
# from feature_selection import top_gene_selection

class GSE144240Dataset(torch.utils.data.Dataset):
    def __init__(self, path='/newdata/why/GSE144240/',mode='train',selected_genes=None):
        # selected_genes=top_gene_selection(path,2000)
        if selected_genes==None:
            selected_genes=np.load('Breast_Result_Gene_GSE144240.npy',allow_pickle=True).tolist()

        self.mode=mode
        self.wsi_imgs=[]
        self.st_coords=[]
        self.st_feats=[]
        self.st_spots_pixel_map=[]
        self.transform=T.Compose([
            T.Resize((224,224)),
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            T.RandomRotation(degrees=(0, 360),fill=255),
            T.ToTensor()]
            )
        self.transform2=T.Compose([
            T.Resize((224,224)),
            T.ToTensor()]
            )

        wsi_filenames=sorted(glob.glob(path+'/*.jpg'))
        random.seed(1553)
        random.shuffle(wsi_filenames)
        random.seed()

        train_frac, val_frac, test_frac = 0.7, 0.0, 0.3
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
            # st_coords=path+'/spot-selections/'+wsi_basename+'_selection.tsv' # stores label, useless in SR
            st_feats=wsi_basename+'_stdata.tsv'
            st_spots_pixel_map=wsi_basename.split(r'_P')[0]+'_spot_data-selection-P'+wsi_basename.split(r'_P')[1]+'.tsv'

            feats_all=pandas.read_csv(st_feats,sep='\t',index_col=0,header=0)
            spots_pixel_map_all=pandas.read_csv(st_spots_pixel_map,sep='\t',header=0)

            for spots_pixel_map in spots_pixel_map_all.iterrows():
                idx_x=int(spots_pixel_map[1]['x'])
                idx_y=int(spots_pixel_map[1]['y'])
                idx=str(idx_x)+'x'+str(idx_y)

                x=spots_pixel_map[1]['pixel_x']
                y=spots_pixel_map[1]['pixel_y']

                try:
                    patch2=wsi_img.crop((int(x-448), int(y-448),int(x+448),int(y+448)))
                    if patch2.size[0]!=896 or patch2.size[1]!=896:
                        print(idx,':',patch2.size)
                        raise Exception
                    
                    patch1=wsi_img.crop((int(x-224), int(y-224),int(x+224),int(y+224)))
                    if patch1.size[0]!=448 or patch1.size[1]!=448:
                        print(idx,':',patch1.size)
                        raise Exception

                    patch=wsi_img.crop((int(x-112), int(y-112),int(x+112),int(y+112)))
                    if patch.size[0]!=224 or patch.size[1]!=224:
                        print(idx,':',patch.size)
                        raise Exception
                except Exception as e:
                    print('patch shape error encountered:',e, idx)
                    continue

                try:
                    feats=np.array(feats_all.loc[idx][selected_genes])
                except Exception as e:
                    print('feats error encountered:',e, idx)
                    continue
                spot_sum=np.sum(feats)
                if spot_sum==0:
                    print('spot_sum=0. skipped:',idx)
                    continue
                feats=np.log1p(feats*1000000/spot_sum)
                self.st_spots_pixel_map.append([idx,x,y])
                self.st_feats.append(np.array(feats))
                # self.wsi_imgs.append({'img':wsi_file,'coord':(x,y)})
                if self.mode=='test':
                    patch=self.transform2(patch)
                    patch1=self.transform2(patch1)
                    patch2=self.transform2(patch2)
                    self.wsi_imgs.append([patch,patch1,patch2])
                else:
                    patch=self.transform(patch)
                    patch1=self.transform(patch1)
                    patch2=self.transform(patch2)
                    self.wsi_imgs.append([patch,patch1,patch2])

        print('total number of samples:', len(self.st_spots_pixel_map))
    def __len__(self):
        return len(self.st_feats)
    
    def __getitem__(self,index):
        # if self.mode=='test':
        #     return self.st_spots_pixel_map[index], self.transform2(self.wsi_imgs[index]), self.st_feats[index]
        # patch_info=self.wsi_imgs[index]
        # x,y=patch_info['coord']
        # patch=cv2.imread(patch_info['img'])[int(x-112):int(x+112), int(y-112):int(y+112),:]
            
        return self.st_spots_pixel_map[index], self.wsi_imgs[index], self.st_feats[index]


    #     print('total number of samples:', len(self.st_spots_pixel_map))

    # def __len__(self):
    #     return len(self.st_feats)
    
    # def __getitem__(self,index):
    #     if self.mode=='test':
    #         return self.st_spots_pixel_map[index], self.transform2(self.wsi_imgs[index]), self.st_feats[index]
    #     # patch_info=self.wsi_imgs[index]
    #     # x,y=patch_info['coord']
    #     # patch=cv2.imread(patch_info['img'])[int(x-112):int(x+112), int(y-112):int(y+112),:]
            
    #     return self.st_spots_pixel_map[index], self.transform(self.wsi_imgs[index]), self.st_feats[index]
# class LRBreastSTDataset(torch.utils.data.Dataset):
#     def __init__(self, path='/newdata/why/Human_breast_cancer_in_situ_capturing_transcriptomics/BRCA',mode='train',selected_genes=None):
#         # selected_genes=top_gene_selection(path,2000)
#         if selected_genes==None: 
#             selected_genes=np.load('Breast_Result_Gene.npy',allow_pickle=True).tolist()
        
#         self.wsi_imgs=[]
#         self.st_coords=[]
#         self.st_feats=[]
#         self.hr_img_gts=[]
#         self.hr_st_feats_gts=[]
#         self.st_spots_pixel_map=[]

#         wsi_filenames=sorted(glob.glob(path+'/*/*.jpg'))#[:5]
#         random.seed(1553)
#         random.shuffle(wsi_filenames)
#         random.seed()

#         train_frac, val_frac, test_frac = 0.7, 0, 0.3
#         n_train=int(len(wsi_filenames)*train_frac) 
#         n_val = int(len(wsi_filenames)*val_frac)
#         n_test=int(len(wsi_filenames)*test_frac)

#         if mode=='train':
#             wsi_filenames=wsi_filenames[:n_train]
#         elif mode=='val':
#             wsi_filenames=wsi_filenames[n_train:n_train+n_val]
#         elif mode=='test':
#             wsi_filenames=wsi_filenames[n_train+n_val:]

#         if mode=='train' or mode=='val':
#             for wsi_file in wsi_filenames:
#                 print('processing:',wsi_file)
#                 wsi_img=cv2.imread(wsi_file)
#                 # wsi_img=cv2.resize(wsi_img, fx=0.5, fy=0.5)

#                 wsi_basename=wsi_file.split(r'.')[0]
#                 st_coords=wsi_basename+'_Coord.tsv' # stores label, useless in SR
#                 st_feats=wsi_basename+'.tsv'
#                 st_spots_pixel_map=wsi_basename+'.spots.txt'

#                 feats_all=pandas.read_csv(st_feats,sep='\t',index_col=0,header=0)
#                 spots_pixel_map_all=pandas.read_csv(st_spots_pixel_map,sep=',',index_col=0,header=0)

#                 for spots_pixel_map in spots_pixel_map_all.iterrows():
#                     idx=spots_pixel_map[0]
#                     idx_x,idx_y=int(idx.split('x')[0]),int(idx.split('x')[1])
#                     x=spots_pixel_map[1]['X']
#                     y=spots_pixel_map[1]['Y']
#                     idx_r=str(idx_x)+'x'+str(idx_y+1)
#                     idx_d=str(idx_x+1)+'x'+str(idx_y)
#                     idx_rd=str(idx_x+1)+'x'+str(idx_y+1)

#                     try:
#                         hr_img=wsi_img[int(x-112):int(x+112+224), int(y-112):int(y+112+224),:]
#                         lr_img=cv2.resize(hr_img,None,fx=0.5, fy=0.5)
#                         if lr_img.shape[0] !=224 or lr_img.shape[1] !=224 or hr_img.shape[0] !=448 or hr_img.shape[1] !=448:
#                             raise Exception
#                     except Exception as e:
#                         print('cropping error encountered:',e,'for',wsi_file)
#                         continue
                    
#                     feats1=np.zeros((1,1))
#                     try:
#                         feats1=np.array(feats_all.loc[idx][selected_genes])
#                     except Exception as e:
#                         print('feats1 error encountered:',e, idx)
#                         continue

#                     self.wsi_imgs.append(lr_img)
#                     self.hr_img_gts.append(hr_img)


#                     # self.hr_img_gts.append(np.array([wsi_img[int(x-112):int(x+112), int(y-112):int(y+112),:], 
#                     #                                  wsi_img[int(x+224-112):int(x+224+112), int(y-112):int(y+112),:], 
#                     #                                  wsi_img[int(x-112):int(x+112), int(y+224-112):int(y+224+112),:], 
#                     #                                  wsi_img[int(x+224-112):int(x+224+112), int(y+224-112):int(y+224+112),:]]))

#                     feats2=np.zeros_like(feats1)
#                     feats3=np.zeros_like(feats1)
#                     feats4=np.zeros_like(feats1)

#                     feats=feats1.copy()

#                     try:
#                         feats2=np.array(feats_all.loc[idx_r][selected_genes])
#                         feats=feats+feats2
#                     except Exception as e:
#                         print('error encountered: feats2.',e)

#                     try:
#                         feats3=np.array(feats_all.loc[idx_d][selected_genes])
#                         feats=feats+feats3
#                     except Exception as e:
#                         print('error encountered: feats3.',e)

#                     try:
#                         feats4=np.array(feats_all.loc[idx_rd][selected_genes])
#                         feats=feats+feats4
#                     except Exception as e:
#                         print('error encountered: feats4.',e)

#                     self.hr_st_feats_gts.append(np.array([feats1, feats2, feats3, feats4]))

#                     self.st_spots_pixel_map.append([idx,x+112,y+112])
#                     self.st_feats.append(feats)

#         elif mode=='test':
#             for wsi_file in wsi_filenames:
#                 tmp_hr_img=[]
#                 tmp_lr_img=[]
#                 tmp_hr_st=[]
#                 tmp_lr_st=[]
#                 tmp_st_spots_pixel_map=[]
#                 print('processing:',wsi_file)
#                 wsi_img=cv2.imread(wsi_file)
#                 # wsi_img=cv2.resize(wsi_img, fx=0.5, fy=0.5)

#                 wsi_basename=wsi_file.split(r'.')[0]
#                 st_coords=wsi_basename+'_Coord.tsv' # stores label, useless in SR
#                 st_feats=wsi_basename+'.tsv'
#                 st_spots_pixel_map=wsi_basename+'.spots.txt'

#                 feats_all=pandas.read_csv(st_feats,sep='\t',index_col=0,header=0)
#                 spots_pixel_map_all=pandas.read_csv(st_spots_pixel_map,sep=',',index_col=0,header=0)

#                 for spots_pixel_map in spots_pixel_map_all.iterrows():
#                     idx=spots_pixel_map[0]
#                     idx_x,idx_y=int(idx.split('x')[0]),int(idx.split('x')[1])
#                     x=spots_pixel_map[1]['X']
#                     y=spots_pixel_map[1]['Y']
#                     idx_r=str(idx_x)+'x'+str(idx_y+1)
#                     idx_d=str(idx_x+1)+'x'+str(idx_y)
#                     idx_rd=str(idx_x+1)+'x'+str(idx_y+1)

#                     try:
#                         hr_img=wsi_img[int(x-112):int(x+112+224), int(y-112):int(y+112+224),:]
#                         lr_img=cv2.resize(hr_img,None,fx=0.5, fy=0.5)
#                         if lr_img.shape[0] !=224 or lr_img.shape[1] !=224 or hr_img.shape[0] !=448 or hr_img.shape[1] !=448:
#                             raise Exception
#                     except Exception as e:
#                         print('cropping error encountered:',e,'for',wsi_file)
#                         continue
                    
#                     feats1=np.zeros((1,1))
#                     try:
#                         feats1=np.array(feats_all.loc[idx][selected_genes])
#                     except Exception as e:
#                         print('feats1 error encountered:',e, idx)
#                         continue

#                     tmp_lr_img.append(lr_img)
#                     tmp_hr_img.append(hr_img)


#                     # self.hr_img_gts.append(np.array([wsi_img[int(x-112):int(x+112), int(y-112):int(y+112),:], 
#                     #                                  wsi_img[int(x+224-112):int(x+224+112), int(y-112):int(y+112),:], 
#                     #                                  wsi_img[int(x-112):int(x+112), int(y+224-112):int(y+224+112),:], 
#                     #                                  wsi_img[int(x+224-112):int(x+224+112), int(y+224-112):int(y+224+112),:]]))

#                     feats2=np.zeros_like(feats1)
#                     feats3=np.zeros_like(feats1)
#                     feats4=np.zeros_like(feats1)

#                     feats=feats1.copy()

#                     try:
#                         feats2=np.array(feats_all.loc[idx_r][selected_genes])
#                         feats=feats+feats2
#                     except Exception as e:
#                         print('error encountered: feats2.',e)

#                     try:
#                         feats3=np.array(feats_all.loc[idx_d][selected_genes])
#                         feats=feats+feats3
#                     except Exception as e:
#                         print('error encountered: feats3.',e)

#                     try:
#                         feats4=np.array(feats_all.loc[idx_rd][selected_genes])
#                         feats=feats+feats4
#                     except Exception as e:
#                         print('error encountered: feats4.',e)

#                     tmp_hr_st.append(np.array([feats1, feats2, feats3, feats4]))

#                     tmp_st_spots_pixel_map.append([idx,x+112,y+112])
#                     tmp_lr_st.append(feats)
                    
#                 self.wsi_imgs.append(tmp_lr_img)
#                 self.hr_st_feats_gts.append(tmp_hr_st)

#                 self.st_spots_pixel_map.append(tmp_st_spots_pixel_map)
#                 self.st_feats.append(tmp_lr_st)
#                 self.hr_img_gts.append(tmp_hr_img)
                
#     def __len__(self):
#         return len(self.st_feats)
    
#     def __getitem__(self,index):
#         return self.st_spots_pixel_map[index], self.wsi_imgs[index], self.st_feats[index], self.hr_img_gts[index], self.hr_st_feats_gts[index]

if __name__=="__main__":
    testdataset=HER2Dataset()
