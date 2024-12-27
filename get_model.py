import torch
import torchvision.models as models
from multi_model import EffNetV2LSwinV2B
from multi_model import EffNetV2LKANSwinV2B
from multi_model import SwinV2BMK
from multi_model import EffSwinKAT
from multi_model import KAN2EffNetV2LSwinV2B
import torchxrayvision as xrv

def get_model(config):
    if config["model"]=="resnet18":
        #print("loading resnet-18")
        model = models.resnet18(pretrained=config['imgnet_pretrained'])
        num_ftrs = model.fc.in_features
        if config['SimCLR_pretraining']==True:
            model.fc = torch.nn.Linear(num_ftrs,100)
        if config['tcl_pretraining']==True:
            model.fc = torch.nn.Linear(num_ftrs,config['projection_head'])
        elif config['SimCLR_pretraining']==False:
            model.fc = torch.nn.Linear(num_ftrs,1)
        model.conv1=torch.nn.Conv2d(in_channels=1,out_channels=64, kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
        model=model.to(config['device'])
    elif config['model']=='convnextl':
        model=models.convnext_large(pretrained=config['imgnet_pretrained'])
        model.features[0][0]=torch.nn.Conv2d(1, 192, kernel_size=(4, 4), stride=(4, 4))
        num_ftrs=model.classifier[2].in_features
        if config['normalized_output']==True:
            model.classifier[2]=torch.nn.Sequential(
                torch.nn.Linear(num_ftrs,1),
                torch.nn.Sigmoid()
            )
        if config['normalized_output']==False:
            model.classifier[2]=torch.nn.Linear(num_ftrs,1)

        model=model.to(config['device'])
    elif config["model"]=="resnet50":
        ##print("loading resnet-18")
        if config['xrv_pretrained']==True:
            model=xrv.models.ResNet(weights="resnet50-res512-all")
            model=model.model
            model.fc=torch.nn.Sequential(
                    torch.nn.Linear(2048, 512, bias=True),
                    torch.nn.Linear(512, 256, bias=True),
                    torch.nn.Linear(256, 1, bias = True))
            model=model.to(config['device'])
        else:        
            model = models.resnet50(pretrained=config['imgnet_pretrained'])
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs,1)
            model.conv1=torch.nn.Conv2d(in_channels=1,out_channels=64, kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
            model=model.to(config['device'])
    elif config["model"]=="effnetv2s":
        model=models.efficientnet_v2_s("EfficientNet_V2_S_Weights.IMAGENET1K_V1")
        model.features[0][0]=torch.nn.Conv2d(in_channels=1,out_channels=24, kernel_size=(3,3),stride=(2,2),padding=(1,1),bias=False)
        num_ftrs=model.classifier[1].in_features 
        model.classifier[1] = torch.nn.Linear(num_ftrs,1)
        model=model.to(config['device'])
    elif config["model"]=="effnetv2m":
        model=models.efficientnet_v2_m("EfficientNet_V2_M_Weights.IMAGENET1K_V1")
        model.features[0][0]=torch.nn.Conv2d(in_channels=1,out_channels=24, kernel_size=(3,3),stride=(2,2),padding=(1,1),bias=False)
        num_ftrs=model.classifier[1].in_features 
        model.classifier[1] = torch.nn.Linear(num_ftrs,1)
        model=model.to(config['device'])   
        
    elif config["model"]=="effnetv2l":
        model=models.efficientnet_v2_l(pretrained=config['imgnet_pretrained'])
        if config['multi_channel_input']==True:
            model.features[0][0]=torch.nn.Conv2d(in_channels=2,out_channels=32, kernel_size=(3,3),stride=(2,2),padding=(1,1),bias=False)
        else:
            model.features[0][0]=torch.nn.Conv2d(in_channels=1,out_channels=32, kernel_size=(3,3),stride=(2,2),padding=(1,1),bias=False)
        
        if config['SimCLR_pretraining']==False and config['tcl_pretraining']==False:
            # In case of config['SimCLR_pretraining']=True model will return 1000 class as the embedding      
            num_ftrs=model.classifier[1].in_features 
            model.classifier[1] = torch.nn.Linear(num_ftrs,1)
        if config['tcl_pretraining']==True:
            num_ftrs=model.classifier[1].in_features 
            model.classifier[1] = torch.nn.Linear(num_ftrs,config['projection_head'])
        model=torch.nn.DataParallel(model)
        model=model.to(config['device'])  
    elif config["model"]=="effnetb4":
        model=models.efficientnet_b4(pretrained=config['imgnet_pretrained']) 
        model.features[0][0]=torch.nn.Conv2d(1, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        num_ftrs=model.classifier[1].in_features 
        model.classifier[1]=torch.nn.Linear(num_ftrs,1)
        model=model.to(config['device'])

    elif config["model"]=="vit_b32":
        model=models.vit_b_32(pretrained=config['imgnet_pretrained'])
        if config['multi_channel_input']==True:
            model.conv_proj=torch.nn.Conv2d(in_channels=2,out_channels=768, kernel_size=(32,32),stride=(32,32))
        else:
            model.conv_proj=torch.nn.Conv2d(in_channels=1,out_channels=768, kernel_size=(32,32),stride=(32,32))
        num_ftrs=model.heads.head.in_features
        model.heads.head = torch.nn.Linear(num_ftrs,1)
        model=model.to(config['device'])
    elif config["model"] == "vit_l16":
        import torch.nn.functional as F

        model = models.vit_l_16(pretrained=config['imgnet_pretrained'])

        # Set the expected image size to 512
        model.image_size = 512

        if config['multi_channel_input'] == True:
            model.conv_proj = torch.nn.Conv2d(in_channels=2, out_channels=1024, kernel_size=(16, 16), stride=(16, 16))
        else:
            model.conv_proj = torch.nn.Conv2d(in_channels=1, out_channels=1024, kernel_size=(16, 16), stride=(16, 16))

        num_ftrs = model.heads.head.in_features
        model.heads.head = torch.nn.Linear(num_ftrs, 1)

        # Adjust positional embeddings for ViT-L/16
        num_patches = (512 // 16) * (512 // 16)  # 32x32 patches for 512x512 images
        original_num_patches = model.encoder.pos_embedding.shape[1] - 1  # Exclude class token

        if num_patches != original_num_patches:
            # Separate class token and positional embeddings
            cls_token = model.encoder.pos_embedding[:, 0:1, :]
            pos_embed = model.encoder.pos_embedding[:, 1:, :]

            # Reshape and interpolate positional embeddings
            dim = pos_embed.shape[-1]
            h_old = w_old = int(original_num_patches ** 0.5)
            h_new = w_new = int(num_patches ** 0.5)

            pos_embed = pos_embed.reshape(1, h_old, w_old, dim).permute(0, 3, 1, 2)
            pos_embed = F.interpolate(pos_embed, size=(h_new, w_new), mode='bilinear', align_corners=False)
            pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, h_new * w_new, dim)

            # Combine class token and new positional embeddings
            new_pos_embed = torch.cat((cls_token, pos_embed), dim=1)
            # Wrap the new positional embeddings in torch.nn.Parameter
            model.encoder.pos_embedding = torch.nn.Parameter(new_pos_embed)

        model = model.to(config['device'])
        
    elif config["model"]=="swin_v2_b":
        model=models.swin_v2_b(pretrained=config['imgnet_pretrained'])
        if config['multi_channel_input']==True:
            model.features[0][0]=torch.nn.Conv2d(2, 128, kernel_size=(4, 4), stride=(4, 4))
        else:
            model.features[0][0]=torch.nn.Conv2d(1, 128, kernel_size=(4, 4), stride=(4, 4))
        num_ftrs=model.head.in_features
        model.head = torch.nn.Linear(num_ftrs,1)
        model=torch.nn.DataParallel(model)
        model=model.to(config['device'])
    elif config["model"]=="swin_t":
        model=models.swin_t(pretrained=config['imgnet_pretrained'])
        if config['multi_channel_input']==True:
            model.features[0][0]=torch.nn.Conv2d(2, 128, kernel_size=(4, 4), stride=(4, 4))
        else:
            model.features[0][0]=torch.nn.Conv2d(1, 128, kernel_size=(4, 4), stride=(4, 4))
        num_ftrs=model.head.in_features
        model.head = torch.nn.Linear(num_ftrs,1)
        model=torch.nn.DataParallel(model)
        model=model.to(config['device'])
    elif config["model"]=="swin_v2_t":
        model=models.swin_v2_t(pretrained=config['imgnet_pretrained'])
        if config['multi_channel_input']==True:
            model.features[0][0]=torch.nn.Conv2d(2, 128, kernel_size=(4, 4), stride=(4, 4))
        else:
            model.features[0][0]=torch.nn.Conv2d(1, 128, kernel_size=(4, 4), stride=(4, 4))
        num_ftrs=model.head.in_features
        model.head = torch.nn.Linear(num_ftrs,1)
        model=torch.nn.DataParallel(model)
        model=model.to(config['device'])
 
    elif config["model"]=="EffNetV2LSwinV2B":
        print("Training w/ EffNetV2LSwinV2B")
        model=EffNetV2LSwinV2B(config)
        model=torch.nn.DataParallel(model)
        model=model.to(config['device'])
    elif config["model"]=="EffNetV2LKANSwinV2B":
        print("Training w/ EffNetV2LKANSwinV2B")
        model=EffNetV2LKANSwinV2B(config)
        # model=torch.nn.DataParallel(model)
        model=model.to(config['device'])
    elif config["model"]=="SwinV2BMK":
        print("Training w/ SwinV2BMK")
        model=SwinV2BMK(config)
        # model=torch.nn.DataParallel(model)
        model=model.to(config['device'])
    elif config["model"]=="EffSwinKAT":
        print("Training w/ EffSwinKAT")
        model=EffSwinKAT(config)
        # model=torch.nn.DataParallel(model)
        model=model.to(config['device'])
    elif config["model"]=="KAN2EffNetV2LSwinV2B":
        print("Training w/ KAN2EffNetV2LSwinV2B")
        model=KAN2EffNetV2LSwinV2B(config)
        # model=torch.nn.DataParallel(model)
        model=model.to(config['device'])

    return model
    