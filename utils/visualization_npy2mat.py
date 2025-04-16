import numpy as np

from scipy.io import loadmat, savemat
import glob
import os
import cv2
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

'''
This file is for visualization in Matlab. Please make sure you have Image Processing Toolbox installed with your Matlab. 
To visualize the 250-channel output, you will need the HyperSpectral Library installed in Image Processing Toolbox.
'''
n_components=1
pca=PCA(n_components=n_components)

gt_npy_files=sorted(glob.glob('/your/path/to/*/*_gt250.npy'))
m2ort_npy_files=sorted(glob.glob('/your/path/to/*/*_m2ort250.npy'))
resnet_npy_files=sorted(glob.glob('/your/path/to/*/*_resnet250.npy'))
vit_npy_files=sorted(glob.glob('/your/path/to/*/*_vit250.npy'))
histogene_npy_files=sorted(glob.glob('/your/path/to/*/*_histogene250.npy'))
hist2st_npy_files=sorted(glob.glob('/your/path/to/*/*_hist2st250.npy'))
stnet_npy_files=sorted(glob.glob('/your/path/to/*/*_stnet250.npy'))

for i in range(len(gt_npy_files)):
    gt_npy=np.load(gt_npy_files[i])
    m2ort_npy=np.load(m2ort_npy_files[i])
    resnet_npy=np.load(resnet_npy_files[i])
    vit_npy=np.load(vit_npy_files[i])
    stnet_npy=np.load(stnet_npy_files[i])
    histogene_npy=np.load(histogene_npy_files[i])
    hist2st_npy=np.load(hist2st_npy_files[i])

    gt_npy=np.reshape(gt_npy,(35*35,250))
    m2ort_npy=np.reshape(m2ort_npy,(35*35,250))
    resnet_npy=np.reshape(resnet_npy,(35*35,250))
    vit_npy=np.reshape(vit_npy,(35*35,250))
    stnet_npy=np.reshape(stnet_npy,(35*35,250))
    histogene_npy=np.reshape(histogene_npy,(35*35,250))
    hist2st_npy=np.reshape(hist2st_npy,(35*35,250))

    pca = pca.fit(gt_npy)

    mse_m2ort=np.sum((m2ort_npy-gt_npy)**2)
    mse_resnet=np.sum((resnet_npy-gt_npy)**2)
    mse_vit=np.sum((vit_npy-gt_npy)**2)
    mse_stnet=np.sum((stnet_npy-gt_npy)**2)
    mse_histogene=np.sum((histogene_npy-gt_npy)**2)
    mse_hist2st=np.sum((hist2st_npy-gt_npy)**2)

    if mse_m2ort<mse_hist2st and mse_m2ort<mse_stnet:
        fname=os.path.basename(gt_npy_files[i]).split('_gt250')[0]
        if not os.path.exists('/your/path/to/selected_for_vis/'+fname):
            os.makedirs('/your/path/to/selected_for_vis/'+fname)
        savemat('/your/path/to/selected_for_vis/'+fname+'/result_'+str(mse_m2ort)+'_m2ort.mat',{'m2ort': m2ort_npy})
        # savemat('/your/path/to/selected_for_vis/'+fname+'/result_'+str(mse_resnet)+'_resnet.mat',{'resnet': resnet_npy})
        # savemat('/your/path/to/selected_for_vis/'+fname+'/result_'+str(mse_vit)+'_vit.mat',{'vit': vit_npy})
        # savemat('/your/path/to/selected_for_vis/'+fname+'/result_'+str(mse_stnet)+'_stnet.mat',{'stnet': stnet_npy})
        # savemat('/your/path/to/selected_for_vis/'+fname+'/result_'+str(mse_histogene)+'_histogene.mat',{'histogene': histogene_npy})
        # savemat('/your/path/to/selected_for_vis/'+fname+'/result_'+str(mse_hist2st)+'_hist2st.mat',{'hist2st': hist2st_npy})
        # savemat('/your/path/to/selected_for_vis/'+fname+'/result_'+str(0)+'_gt.mat',{'gt': gt_npy})
        
        
        # gt_npy=np.reshape(pca.transform(gt_npy),(35,35,n_components))
        # m2ort_npy=np.reshape(pca.transform(m2ort_npy),(35,35,n_components))
        # resnet_npy=np.reshape(pca.transform(resnet_npy),(35,35,n_components))
        # vit_npy=np.reshape(pca.transform(vit_npy),(35,35,n_components))
        # stnet_npy=np.reshape(pca.transform(stnet_npy),(35,35,n_components))
        # histogene_npy=np.reshape(pca.transform(histogene_npy),(35,35,n_components))
        # hist2st_npy=np.reshape(pca.transform(hist2st_npy),(35,35,n_components))

        gt_npy=np.reshape(pca.transform(gt_npy),(35,35))
        m2ort_npy=np.reshape(pca.transform(m2ort_npy),(35,35))
        resnet_npy=np.reshape(pca.transform(resnet_npy),(35,35))
        vit_npy=np.reshape(pca.transform(vit_npy),(35,35))
        stnet_npy=np.reshape(pca.transform(stnet_npy),(35,35))
        histogene_npy=np.reshape(pca.transform(histogene_npy),(35,35))
        hist2st_npy=np.reshape(pca.transform(hist2st_npy),(35,35))

        gt_npy=cv2.resize(gt_npy,(140,140),interpolation=cv2.INTER_NEAREST_EXACT)
        m2ort_npy=cv2.resize(m2ort_npy,(140,140),interpolation=cv2.INTER_NEAREST_EXACT)
        resnet_npy=cv2.resize(resnet_npy,(140,140),interpolation=cv2.INTER_NEAREST_EXACT)
        vit_npy=cv2.resize(vit_npy,(140,140),interpolation=cv2.INTER_NEAREST_EXACT)
        stnet_npy=cv2.resize(stnet_npy,(140,140),interpolation=cv2.INTER_NEAREST_EXACT)
        histogene_npy=cv2.resize(histogene_npy,(140,140),interpolation=cv2.INTER_NEAREST_EXACT)
        hist2st_npy=cv2.resize(hist2st_npy,(140,140),interpolation=cv2.INTER_NEAREST_EXACT)

        plt.imsave('/your/path/to/selected_for_vis/'+fname+'/result_'+str(mse_m2ort)+'_m2ort.jpg',m2ort_npy,cmap='jet')
        # plt.imsave('/your/path/to/selected_for_vis/'+fname+'/result_'+str(mse_resnet)+'_resnet.jpg',resnet_npy,cmap='jet')
        # plt.imsave('/your/path/to/selected_for_vis/'+fname+'/result_'+str(mse_vit)+'_vit.jpg',vit_npy,cmap='jet')
        # plt.imsave('/your/path/to/selected_for_vis/'+fname+'/result_'+str(mse_stnet)+'_stnet.jpg',stnet_npy,cmap='jet')
        # plt.imsave('/your/path/to/selected_for_vis/'+fname+'/result_'+str(mse_histogene)+'_histogene.jpg',histogene_npy,cmap='jet')
        # plt.imsave('/your/path/to/selected_for_vis/'+fname+'/result_'+str(mse_hist2st)+'_hist2st.jpg',hist2st_npy,cmap='jet')
        # plt.imsave('/your/path/to/selected_for_vis/'+fname+'/result_'+str(0)+'_gt.jpg',gt_npy,cmap='jet')