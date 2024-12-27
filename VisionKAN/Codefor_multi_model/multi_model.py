import torch
import torchvision.models as models
from VisionKAN.models_kan import kanBlock, VisionKAN


  # Adjust the import based on the actual file structure

class EffNetV2LSwinV2B(torch.nn.Module):
    def __init__(self,config):
        super().__init__()
        self.effnetv2l=models.efficientnet_v2_l(pretrained=config['imgnet_pretrained'])
        self.effnetv2l.features[0][0]=torch.nn.Conv2d(in_channels=1,out_channels=32, kernel_size=(3,3),stride=(2,2),padding=(1,1),bias=False)
        # self.effnetv2l.classifier[1] = torch.nn.Identity()
        self.num_ftrs_effnetv2l=self.effnetv2l.classifier[1].in_features 
        
        self.effnetv2l.classifier[1] = torch.nn.Linear(self.num_ftrs_effnetv2l,1)
        # self.effnetv2l=torch.nn.DataParallel(self.effnetv2l)
        self.eff_state_dict=torch.load('saved_models/effnetv2l_5_1')
        new_state_dict = {}
        for k, v in self.eff_state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict
        
        self.effnetv2l.load_state_dict(state_dict)
        self.effnetv2l.classifier[1] = torch.nn.Linear(self.num_ftrs_effnetv2l,4096)
        
                                                 

        self.swinv2b=models.swin_v2_b(pretrained=config['imgnet_pretrained'])
        self.swinv2b.features[0][0]=torch.nn.Conv2d(1, 128, kernel_size=(4, 4), stride=(4, 4))
        # self.swinv2b.head = torch.nn.Identity()
        self.num_ftrs_swinv2b=self.swinv2b.head.in_features
        self.swinv2b.head = torch.nn.Linear(self.num_ftrs_swinv2b,1)
        
        # self.lin=torch.nn.Sequential(torch.nn.Linear(512,64),
        #                              torch.nn.Linear(64,1))

    def forward(self, x):
        x1 = self.effnetv2l(x)
        # x1=x1.detach()
        x1=x1.reshape(-1,1,64,64)
        x2 = self.swinv2b(x1)        
        return x2

class EffNetV2LKANSwinV2B(torch.nn.Module):
    def __init__(self,config):
        super().__init__()
        self.effnetv2l=models.efficientnet_v2_l(pretrained=config['imgnet_pretrained'])
        self.effnetv2l.features[0][0]=torch.nn.Conv2d(in_channels=1,out_channels=32, kernel_size=(3,3),stride=(2,2),padding=(1,1),bias=False)
        # self.effnetv2l.classifier[1] = torch.nn.Identity()
        self.num_ftrs_effnetv2l=self.effnetv2l.classifier[1].in_features 
        
        self.effnetv2l.classifier[1] = torch.nn.Linear(self.num_ftrs_effnetv2l,1)
        # self.effnetv2l=torch.nn.DataParallel(self.effnetv2l)
        self.eff_state_dict=torch.load('saved_models/effnetv2l_5_1')
        new_state_dict = {}
        for k, v in self.eff_state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict
        
        self.effnetv2l.load_state_dict(state_dict)
        # self.effnetv2l.classifier[1] = torch.nn.Linear(self.num_ftrs_effnetv2l,4096)
        
        # SwinV2B part 
        # Load the pretrained SwinV2-B model
        self.swinv2b = models.swin_v2_b(pretrained=config['imgnet_pretrained'])

        # Modify the first Conv2d layer to accept 1-channel input
        self.swinv2b.features[0][0] = torch.nn.Conv2d(1, 128, kernel_size=(4, 4), stride=(4, 4))

        # Replace the MLP block in each SwinTransformerBlockV2 with kanBlock
        for name, module in self.swinv2b.named_modules():
            if isinstance(module, type(self.swinv2b.features[1][0])):  # Checking for SwinTransformerBlockV2 type
                module.mlp = kanBlock(dim=module.mlp[0].in_features, num_heads=8)


        # Adjust the final classification layer
        self.num_ftrs_swinv2b = self.swinv2b.head.in_features
        self.swinv2b.head = torch.nn.Linear(self.num_ftrs_swinv2b, 1)  # Single output for regression


    def forward(self, x):
        x1 = self.effnetv2l(x)
        print(x1.shape)
        # x1=x1.detach()
        x1=x1.reshape(-1,1,64,64)
        print(x1.shape)

        x2 = self.swinv2b(x1)

        return x2




class KAN2EffNetV2LSwinV2B(torch.nn.Module):
    def __init__(self, config): 
        super().__init__()  

        self.effnetv2l = models.efficientnet_v2_l(pretrained=config['imgnet_pretrained']) 
        # Modify first convolution layer to handle 1-channel input 
        self.effnetv2l.features[0][0] = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False) 
        self.num_ftrs_effnetv2l=self.effnetv2l.classifier[1].in_features 
        
        # Replace features[2] with VisionKAN 
        self.effnetv2l.features[2] = VisionKAN(     
            batch_size=4        
        ) 
        # Change the img_size in PatchEmbed to 256x256
        # self.effnetv2l.patch_embed.img_size = (256, 256)

        # print(self.effnetv2l)
        
        self.effnetv2l.classifier[1] = torch.nn.Linear(self.num_ftrs_effnetv2l,1)


        # Optional: Print classifier structure to verify
        # print(self.effnetv2l.classifier)

        self.eff_state_dict=torch.load('saved_models/effnetv2l_5_1')
        new_state_dict = {}
        for k, v in self.eff_state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict
        
        self.effnetv2l.load_state_dict(state_dict, strict=False)

        
        # self.effnetv2l.classifier[1] = torch.nn.Linear(self.num_ftrs_effnetv2l,4096)


        # # Remove classifier weights to avoid size mismatch 
        # if 'classifier.1.weight' in state_dict: 
        #     del state_dict['classifier.1.weight'] 
        # if 'classifier.1.bias' in state_dict: 
        #     del state_dict['classifier.1.bias'] 

        # # Remove weights for features[2] since it has been replaced by VisionKAN 
        # for key in list(state_dict.keys()): 
        #     if key.startswith('features.2'): 
        #         del state_dict[key] 

  

        # # Load the remaining state dict into the model 
        # self.effnetv2l.load_state_dict(state_dict, strict=False) 
        # # Replace classifier with a new linear layer for regression 

        # self.effnetv2l.classifier[1] = torch.nn.Linear(self.num_ftrs_effnetv2l, 1) 
        
        # SwinV2B part remains unchanged
        self.swinv2b = models.swin_v2_b(pretrained=config['imgnet_pretrained'])
        self.swinv2b.features[0][0] = torch.nn.Conv2d(1, 128, kernel_size=(4, 4), stride=(4, 4))

        for name, module in self.swinv2b.named_modules():
            if type(module).__name__ == "SwinTransformerBlockV2":
                module.mlp = kanBlock(dim=module.mlp[0].in_features, num_heads=8)

        # Adjust the final layer for your task
        self.num_ftrs_swinv2b = self.swinv2b.head.in_features
        self.swinv2b.head = torch.nn.Linear(self.num_ftrs_swinv2b, 1)

    def forward(self, x):
        # Step 1: Print the input shape
        print(f"Input shape to EffNetV2L: {x.shape}")  # This should be [4, 1, 512, 512]

        # Step 2: Do not expand the grayscale input to 3 channels since EffNetV2L expects 1 channel
        # Comment out or remove any code that expands the input channels

        # Step 3: Pass the 1-channel input to EfficientNetV2-L
        try:
            x1 = self.effnetv2l(x)
            print(f"Shape after EffNetV2L: {x1.shape}")
        except Exception as e:
            print(f"Error in EffNetV2L forward pass: {e}")
            return None

        # Step 4: Continue with your model pipeline
        x1 = x1.view(-1, 32, 8, 8)  # Adjust reshape based on your architecture
        x2 = self.swinv2b(x1)

        return x2

