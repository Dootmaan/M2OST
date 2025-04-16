import torch
import cv2
from sklearn.decomposition import PCA
import numpy as np
import random
from dataset.HBCDataset import HBCDataset
from model.M2ORT import M2ORT
import pandas
import PIL
import torchvision
import glob

selected_genes=np.load('/your/path/dataset/Breast_Result_Gene.npy',allow_pickle=True).tolist()

model_msvit=M2ORT(num_classes=250).to('cuda')
model_msvit=torch.nn.DataParallel(model_msvit)
model_msvit.load_state_dict(torch.load('/your/path/',map_location='cpu'))
model_resnet50=torchvision.models.resnet50(num_classes=250).to('cuda')
model_resnet50.load_state_dict(torch.load('/your/path/ResNet_best.pth',map_location='cpu'))
model_vitb16=torchvision.models.vit_b_16(num_classes=250)
model_vitb16.heads=torch.nn.Sequential(
    torch.nn.Linear(768,len(selected_genes))
)
model_vitb16.to('cuda')
model_vitb16.load_state_dict(torch.load('/your/path/ViT_best.pth',map_location='cpu'))
model_stnet=torchvision.models.densenet121(num_classes=250).to('cuda')
model_stnet.load_state_dict(torch.load('/your/path/DenseNet_best.pth',map_location='cpu'))

model_msvit.eval()
model_resnet50.eval()
model_vitb16.eval()
model_stnet.eval()

wsi_files=glob.glob('/your/path/Human_breast_cancer_in_situ_capturing_transcriptomics/BRCA/*/*.jpg')
for wsi_file in wsi_files:
    print('processing:', wsi_file)
    wsi_img=PIL.Image.open(wsi_file)
    wsi_basename=wsi_file.split(r'.')[0]
    st_coords=wsi_basename+'_Coord.tsv' # stores label, useless in SR
    st_feats=wsi_basename+'.tsv'
    st_spots_pixel_map=wsi_basename+'.spots.txt'

    feats_all=pandas.read_csv(st_feats,sep='\t',index_col=0,header=0)[selected_genes]
    spots_pixel_map_all=pandas.read_csv(st_spots_pixel_map,sep=',',index_col=0,header=0)

    # pca = PCA(n_components=1)	#实例化
    # pca = pca.fit(feats_all.values)			#拟合模型

    # random.shuffle(selected_genes)
    # selected_genes=selected_genes[:512]

    # model_map = resnet50(num_classes=len(selected_genes)).to('cuda')

    # model_sr = SRUNet(len(selected_genes),len(selected_genes))

    # lr_trainset=LRBreastSTDataset('/your/path/Human_breast_cancer_in_situ_capturing_transcriptomics/BRCA/',mode='train', selected_genes=selected_genes)
    # valset=EmbededFeatsDataset('/your/path/CAMELYON16/',mode='val',level=1)
    # testset=BreastSTDataset('/your/path/Human_breast_cancer_in_situ_capturing_transcriptomics/BRCA/',mode='test', selected_genes=selected_genes)

    # trainloader=torch.utils.data.DataLoader(lr_trainset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    # valloader=torch.utils.data.DataLoader(valset, batch_size=1, shuffle=True, drop_last=False)
    # testloader=torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, drop_last=False)

    preds_msvit=np.zeros((35,35,250))
    preds_resnet=np.zeros((35,35,250))
    preds_vit=np.zeros((35,35,250))
    preds_stnet=np.zeros((35,35,250))

    gts=np.zeros((35,35,250))

    for spots_pixel_map in spots_pixel_map_all.iterrows():
        idx=spots_pixel_map[0]
        x=spots_pixel_map[1]['X']
        y=spots_pixel_map[1]['Y']

        idx_x=int(idx.split('x')[0])
        idx_y=int(idx.split('x')[1])

        try:
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

        patch=torchvision.transforms.ToTensor()(patch).type(torch.FloatTensor).to('cuda').unsqueeze(0)
        with torch.no_grad():
            output_msvit=model_msvit(patch)
            output_resnet50=model_resnet50(patch)
            output_vitb16=model_vitb16(patch)
            output_stnet=model_stnet(patch)


        gts[idx_x,idx_y,:]=feats
        # self.st_spots_pixel_map.append([idx,x,y])
        # self.st_feats.append(np.array(feats))
        # self.wsi_imgs.append(patch)

        preds_msvit[idx_x,idx_y,:]=output_msvit.squeeze(0).cpu().data.numpy()
        preds_resnet[idx_x,idx_y,:]=output_resnet50.squeeze(0).cpu().data.numpy()
        preds_vit[idx_x,idx_y,:]=output_vitb16.squeeze(0).cpu().data.numpy()
        preds_stnet[idx_x,idx_y,:]=output_stnet.squeeze(0).cpu().data.numpy()

        # idx, x, y=spot_index_map

        # lr_img_tensor=lr_img.float().to('cuda').permute(0,3,1,2)
        # lr_st_tensor=lr_st.float().to('cuda')

        # hr_img_tensor=hr_img.float().to('cuda').permute(0,3,1,2)
        # hr_st_reshape=np.reshape(hr_st.squeeze(0).numpy(), (-1,3))
        # hr_st_reshape_new = pca.transform(hr_st_reshape)

        # lr_st_pred_output=model_map(lr_img_tensor)

        # loss_st=st_cri(lr_st_pred_output, lr_st_tensor)
        # optimizer1.zero_grad()
        # loss_st.backward()
        # optimizer1.step()

        # _, output_hr_st, _, _  = model_main(lr_img_tensor, lr_st_tensor)
        # output_hr_st=np.reshape(output_hr_st.squeeze(-1).squeeze(0).numpy(), (-1,3))
        # output_hr_st_new=pca.transform(output_hr_st)

        # idx_x=int(idx.split('x')[0])
        # idx_y=int(idx.split('x')[1])

        # gts[x:x+1,y:y+1,:]=hr_st_reshape_new
        # preds[x:x+1,y:y+1,:]=output_hr_st_new

    # content_area=np.where(gts!=-1)
    # min_x=min(content_area[0])
    # max_x=max(content_area[0])
    # min_y=min(content_area[1])
    # max_y=max(content_area[1])
    # gts=gts[min_x:max_x,min_y:max_y,:]
    # gts[gts<0]=0
    # preds_msvit=preds_msvit[min_x:max_x,min_y:max_y,:]
    # preds_msvit[preds_msvit<0]=0
    # preds_resnet=preds_resnet[min_x:max_x,min_y:max_y,:]
    # preds_resnet[preds_resnet<0]=0
    # preds_vit=preds_vit[min_x:max_x,min_y:max_y,:]
    # preds_vit[preds_vit<0]=0
    # preds_stnet=preds_stnet[min_x:max_x,min_y:max_y,:]
    # preds_stnet[preds_stnet<0]=0

    np.save(wsi_basename+'_gt250.npy', gts)
    np.save(wsi_basename+'_m2ort250.npy',preds_msvit)
    np.save(wsi_basename+'_resnet250.npy',preds_resnet)
    np.save(wsi_basename+'_vit250.npy',preds_vit)
    np.save(wsi_basename+'_stnet250.npy',preds_stnet)
    print('npy saved to ',wsi_basename+'*250.npy')
