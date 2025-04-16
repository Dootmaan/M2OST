import torch
from dataset.HBCDataset import HBCDataset
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
# import torchvision
# from vit_pytorch import SimpleViT
from model.M2OST import M2OST
import random
import math
from scipy.stats import pearsonr
import time
# import torch.distributed as dist

EPOCH=200
LR=1e-4
BATCH_SIZE=48

selected_genes=np.load('/home/why/Workspace-Python/SRofST/dataset/HBC_Selected_Genes.npy',allow_pickle=True).tolist()
# random.seed(1553)
# selected_genes=random.sample(selected_genes,250)
# random.seed()

writer = SummaryWriter(log_dir = './logs')
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model_gen = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
# model_gen.fc=torch.nn.Linear(2048,len(selected_genes))
model_gen=M2OST(num_classes=250,depth=4,dim=192*3, mlp_dim=192*2, heads=9,dim_head=64)
# model_s=ViT(depth=6,dim=192*3, mlp_dim=192*2, heads=9,dim_head=64)
# model_b=ViT()
# model_l=ViT(depth=12,dim=384*3, mlp_dim=384*2, heads=18,dim_head=64)
# model_gen.heads=torch.nn.Sequential(
#     torch.nn.Linear(768,len(selected_genes))
# )
model_gen.to(device)
# model_gen=torch.nn.DataParallel(torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_gen).to(device),device_ids=[0,1])
model_gen=torch.nn.DataParallel(model_gen,device_ids=[0,1])
model_gen.load_state_dict(torch.load('/path/to/M2OST_small_best.pth',map_location='cpu'))
trainset=HBCDataset('/path/to/Human_breast_cancer_in_situ_capturing_transcriptomics/BRCA/',mode='train', selected_genes=selected_genes)
valset=HBCDataset('/path/to/Human_breast_cancer_in_situ_capturing_transcriptomics/BRCA/',mode='val',selected_genes=selected_genes)
testset=HBCDataset('/path/to/Human_breast_cancer_in_situ_capturing_transcriptomics/BRCA/',mode='test', selected_genes=selected_genes)

trainloader=torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
valloader=torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
testloader=torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

# optimizer0=torch.optim.RMSprop(model_gen.parameters(),lr=LR)
optimizer0=torch.optim.Adam(model_gen.parameters(),lr=LR)
# optimizer1=torch.optim.RMSprop(model_dis.parameters(),lr=LR)
# lossfunc_dis=torch.nn.L1Loss()
lossfunc_gen=torch.nn.MSELoss()

def TestModel(valloader):
    time0=time.time()
    model_gen.eval()
    pearson_list=[]
    rmse_list=[]
    for i,data in enumerate(valloader):
        _,img,label=data
        img1=img[0].type(torch.FloatTensor).to(device)
        img2=img[1].type(torch.FloatTensor).to(device)
        img3=img[2].type(torch.FloatTensor).to(device)
        label=label.data.numpy()
        with torch.no_grad():
            output=model_gen(img1,img2,img3)
        output=output.cpu().data.numpy()
        pearson_list.append(pearsonr(output.squeeze(0), label.squeeze(0))[0])
        rmse_list.append(math.sqrt(np.sum((output.squeeze(0)-label.squeeze(0))**2)/len(selected_genes)))
        pvalue=pearsonr(output, label)[1]
        if i%100==0:
            print('[TEST ITER {}/{}] Pearson: {}, p-value: {}'.format(i, len(valloader), pearsonr(output.squeeze(0), label.squeeze(0))[0],pearsonr(output.squeeze(0), label.squeeze(0))[1]))

    print('Avg Pearson:',np.sum(pearson_list)/len(pearson_list))
    print('Avg RMSE:',np.sum(rmse_list)/len(rmse_list))
    print('total time:',time.time()-time0)
    return np.sum(pearson_list)/len(pearson_list)

best_acc=0.1
TestModel(testloader)
# raise Exception

for x in range(EPOCH):
    model_gen.train()
    for i,data in enumerate(trainloader):
        _,img,label=data
        img1=img[0].type(torch.FloatTensor).to(device)
        img2=img[1].type(torch.FloatTensor).to(device)
        img3=img[2].type(torch.FloatTensor).to(device)
        label=label.type(torch.FloatTensor).to(device)
        output=model_gen(img1,img2,img3)

        optimizer0.zero_grad()

        loss_gen=lossfunc_gen(output,label)

        loss_gen.backward()
        optimizer0.step()

        # for p in model_dis.parameters():
        #     p.data.clamp_(-0.01, 0.01)

        if i%10==0:
            print('[',x,i,'] loss:',loss_gen.item())

    acc=TestModel(testloader)
    writer.add_scalar('loss', loss_gen.item(), global_step=x, walltime=None)
    writer.add_scalar('Pearson', acc, global_step=x, walltime=None)
    if acc>best_acc:
        best_acc=acc
        print('New best Pearson:',acc)
        torch.save(model_gen.state_dict(),'M2OST_small_best.pth')
