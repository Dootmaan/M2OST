from thop import clever_format
from thop import profile
import torch
from model.M2ORT import M2ORT
import torchvision

inputs=torch.randn(1,1, 3, 224, 224)
# model=M2ORT(depth=6,dim=192*3, mlp_dim=192*2, heads=9,dim_head=64)
# model=M2ORT(depth=6,dim=192*3, mlp_dim=3072, heads=9,dim_head=64)
# model=M2ORT(dim_head = 64, mlp_dim=3072)
# model=M2ORT(depth=12,dim=384*3, mlp_dim=384*2, heads=18,dim_head=64)
model=M2ORT(depth=12,dim=384*3, mlp_dim=3072, heads=18,dim_head=64)
# model=torchvision.models.vgg16(num_classes=250)
# model=torchvision.models.vit_b_16(num_classes=250)
# model=torchvision.models.resnet50(num_classes=250)
# model=torchvision.models.densenet121(num_classes=250)
# model.fc=torch.nn.Linear(2048,250)
# model.cuda()
flops, params = profile(model, inputs=(inputs))
# print('mam mem:',torch.cuda.max_memory_allocated()/MB)
flops, params = clever_format([flops, params], "%.3f")
print(flops,params)

# Find total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())

print(total_params, 'total parameters.')
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(total_trainable_params,'training parameters.')