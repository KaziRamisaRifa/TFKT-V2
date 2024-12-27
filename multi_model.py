import torch
import torchvision.models as models
from VisionKAN.models_kan import kanBlock, VisionKAN
# from rational_kat_cu.kat_rational import KAT_Group


  # Adjust the import based on the actual file structure

class EffNetV2LSwinV2B(torch.nn.Module):
    def __init__(self,config):
        super().__init__()
        self.effnetv2l=models.efficientnet_v2_l(pretrained=config['imgnet_pretrained'])
        self.effnetv2l.features[0][0]=torch.nn.Conv2d(in_channels=1,out_channels=32, kernel_size=(3,3),stride=(2,2),padding=(1,1),bias=False)
        # self.effnetv2l.classifier[1] = torch.nn.Identity()
        self.num_ftrs_effnetv2l=self.effnetv2l.classifier[1].in_features 

        # Important Line below
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
        
                                                 
        # config['imgnet_pretrained'] = False
        self.swinv2b=models.swin_v2_b(pretrained=config['imgnet_pretrained'])
        self.swinv2b.features[0][0]=torch.nn.Conv2d(1, 128, kernel_size=(4, 4), stride=(4, 4))
        # self.swinv2b.head = torch.nn.Identity()
        self.num_ftrs_swinv2b=self.swinv2b.head.in_features
        self.swinv2b.head = torch.nn.Linear(self.num_ftrs_swinv2b,1)
        
        # self.lin=torch.nn.Sequential(torch.nn.Linear(512,64),
        #                              torch.nn.Linear(64,1))

    def forward(self, x):
        # print(f"Before eff: {x.shape}")
        x1 = self.effnetv2l(x)
        # print(f"After eff: {x1.shape}")
        # x1=x1.detach()
        # Important Line below
        x1=x1.reshape(-1,1,64,64)
        x2 = self.swinv2b(x)        
        return x2



class EffNetV2LKANSwinV2B(torch.nn.Module):
    def __init__(self,config):
        super().__init__()
        # config['imgnet_pretrained'] = False 
        self.effnetv2l=models.efficientnet_v2_l(pretrained=config['imgnet_pretrained'])
        self.effnetv2l.features[0][0]=torch.nn.Conv2d(in_channels=1,out_channels=32, kernel_size=(3,3),stride=(2,2),padding=(1,1),bias=False)
        # self.effnetv2l.classifier[1] = torch.nn.Identity()
        self.num_ftrs_effnetv2l=self.effnetv2l.classifier[1].in_features 

        # print("\nok")
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
        
        # config['imgnet_pretrained'] = False                                        
        self.swinv2b = models.swin_v2_b(pretrained=config['imgnet_pretrained'])
        self.swinv2b.features[0][0] = torch.nn.Conv2d(1, 128, kernel_size=(4, 4), stride=(4, 4))

        for name, module in self.swinv2b.named_modules():
            if hasattr(module, 'mlp'):
                # Check the structure of module.mlp
                # print(f"Inspecting {name}: mlp structure:")
                # print(dir(module.mlp))  # Print out the attributes to understand the structure

                # If it's not iterable, you may need to access layers by specific attributes
                # For example, if module.mlp is a class with specific named layers:
                if hasattr(module.mlp, 'linear_layers'):
                    layers = module.mlp.linear_layers  # Assuming 'linear_layers' is the attribute holding layers
                    in_features = layers[0].in_features
                    out_features = layers[-1].out_features
                    module.mlp = kanBlock(in_features, out_features)

        # Update the final layer if needed
        self.num_ftrs_swinv2b = self.swinv2b.head.in_features
        self.swinv2b.head = torch.nn.Linear(self.num_ftrs_swinv2b, 1)

        
        # self.lin=torch.nn.Sequential(torch.nn.Linear(512,64),
        #                              torch.nn.Linear(64,1))

    def forward(self, x):
        # print(f"Before eff: {x.shape}")
        x1 = self.effnetv2l(x)
        # print(f"After eff: {x1.shape}")
        # x1=x1.detach()
        x1=x1.reshape(-1,1,64,64)
        x2 = self.swinv2b(x1)        
        return x2




class SwinV2BMK(torch.nn.Module):
    def __init__(self, config):
        super(SwinV2BMK, self).__init__()
                                      
        self.swinv2b = models.swin_v2_b(pretrained=config['imgnet_pretrained'])

        self.swinv2b.features[0][0] = torch.nn.Conv2d(1, 128, kernel_size=(4, 4), stride=(4, 4))

        for name, module in self.swinv2b.named_modules():
            if hasattr(module, 'mlp'):
                # Get input/output feature sizes from MLP
                if hasattr(module.mlp, 'linear_layers'):
                    layers = module.mlp.linear_layers  #'linear_layers' is the attribute holding layers
                    in_features = layers[0].in_features
                    out_features = layers[-1].out_features
                    
                    # Save original MLP and create a kanBlock with the same input/output size
                    self.original_mlp = module.mlp
                    self.kan_block = kanBlock(in_features, out_features)
                    
                    # Replace the MLP with a custom forward method to average the output of both
                    module.mlp = torch.nn.Module()  # Blank out the MLP so we can control the forward pass
                    setattr(module, 'mlp_forward', self.mlp_kan_forward)

        # Update the final layer if needed
        self.num_ftrs_swinv2b = self.swinv2b.head.in_features
        self.swinv2b.head = torch.nn.Linear(self.num_ftrs_swinv2b, 1)

    
    def cross_attention(self, query, key, value):
        # Cross-attention between Query and Key/Value
        attention_scores = torch.matmul(query, key.transpose(-1, -2))  # Shape: [batch_size, seq_len, seq_len]
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)  # Normalize attention weights
        attended_output = torch.matmul(attention_weights, value)  # Shape: [batch_size, seq_len, features]
        
        return attended_output
    
    def mlp_kan_forward(self, x):
        # Compute both MLP and kanBlock outputs
        mlp_output = self.original_mlp(x)
        kan_output = self.kan_block(x)

        # Step 1: Cross-attention with Q = mlp_output, K = V = kan_output
        cross_attention_mlp_kan = self.cross_attention(mlp_output, kan_output, kan_output)
        
        # Step 2: Cross-attention with Q = kan_output, K = V = mlp_output
        cross_attention_kan_mlp = self.cross_attention(kan_output, mlp_output, mlp_output)

        # Step 3: Sum the outputs of the two cross-attention operations
        output = cross_attention_mlp_kan + cross_attention_kan_mlp

        return output

    def forward(self, x):
        # Forward pass through the modified SwinV2B model
        return self.swinv2b(x)



class EffSwinKAT(torch.nn.Module):
    def __init__(self, config):
        super(EffSwinKAT, self).__init__() 
         
        self.effnetv2l=models.efficientnet_v2_l(pretrained=config['imgnet_pretrained'])
        self.effnetv2l.features[0][0]=torch.nn.Conv2d(in_channels=1,out_channels=32, kernel_size=(3,3),stride=(2,2),padding=(1,1),bias=False)
        # self.effnetv2l.classifier[1] = torch.nn.Identity()
        self.num_ftrs_effnetv2l=self.effnetv2l.classifier[1].in_features 

        # print("\nok")
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

        config['imgnet_pretrained'] = False  
        self.swinv2b = models.swin_v2_b(pretrained=config['imgnet_pretrained'])

        self.swinv2b.features[0][0] = torch.nn.Conv2d(1, 128, kernel_size=(4, 4), stride=(4, 4))

        for name, module in self.swinv2b.named_modules():
            if hasattr(module, 'mlp'):
                # Get input/output feature sizes from MLP
                if hasattr(module.mlp, 'linear_layers'):
                    layers = module.mlp.linear_layers  #'linear_layers' is the attribute holding layers
                    in_features = layers[0].in_features
                    out_features = layers[-1].out_features
                    
                    # Save original MLP and create a kanBlock with the same input/output size
                    self.original_mlp = module.mlp
                    self.kan_block = kanBlock(in_features, out_features)
                    
                    # Replace the MLP with a custom forward method to average the output of both
                    module.mlp = torch.nn.Module()  # Blank out the MLP so we can control the forward pass
                    setattr(module, 'mlp_forward', self.mlp_kan_forward)

        # Update the final layer if needed
        self.num_ftrs_swinv2b = self.swinv2b.head.in_features
        self.swinv2b.head = torch.nn.Linear(self.num_ftrs_swinv2b, 1)

    
    def cross_attention(self, query, key, value):
        # Cross-attention between Query and Key/Value
        attention_scores = torch.matmul(query, key.transpose(-1, -2))  # Shape: [batch_size, seq_len, seq_len]
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)  # Normalize attention weights
        attended_output = torch.matmul(attention_weights, value)  # Shape: [batch_size, seq_len, features]
        
        return attended_output
    
    def mlp_kan_forward(self, x):
        # Compute both MLP and kanBlock outputs
        mlp_output = self.original_mlp(x)
        kan_output = self.kan_block(x)

        # Step 1: Cross-attention with Q = mlp_output, K = V = kan_output
        cross_attention_mlp_kan = self.cross_attention(mlp_output, kan_output, kan_output)
        
        # Step 2: Cross-attention with Q = kan_output, K = V = mlp_output
        cross_attention_kan_mlp = self.cross_attention(kan_output, mlp_output, mlp_output)

        # Step 3: Sum the outputs of the two cross-attention operations
        output = cross_attention_mlp_kan + cross_attention_kan_mlp

        return output

    def forward(self, x):
        # print(f"Before eff: {x.shape}")
        x1 = self.effnetv2l(x)
        # print(f"After eff: {x1.shape}")
        # x1=x1.detach()
        x1=x1.reshape(-1,1,64,64)
        x2 = self.swinv2b(x1)        
        return x2




class KAN2EffNetV2LSwinV2B(torch.nn.Module):
    def __init__(self,config):
        super().__init__()
        self.effnetv2l=models.efficientnet_v2_l(pretrained=config['imgnet_pretrained'])
        self.effnetv2l.features[0][0]=torch.nn.Conv2d(in_channels=1,out_channels=32, kernel_size=(3,3),stride=(2,2),padding=(1,1),bias=False)
        # self.effnetv2l.classifier[1] = torch.nn.Identity()
        self.num_ftrs_effnetv2l=self.effnetv2l.classifier[1].in_features 

        # print("\nok")
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
        print(self.num_ftrs_swinv2b)
        
        # self.lin=torch.nn.Sequential(torch.nn.Linear(512,64),
        #                              torch.nn.Linear(64,1))

    def forward(self, x):
        # print(f"Before eff: {x.shape}")
        x1 = self.effnetv2l(x)
        # print(f"After eff: {x1.shape}")
        # x1=x1.detach()
        x1=x1.reshape(-1,1,64,64)
        x2 = self.swinv2b(x1)        
        return x2
















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



# class EffNetV2LKANSwinV2B(torch.nn.Module):
#     def __init__(self,config):
#         super().__init__()
#         self.effnetv2l=models.efficientnet_v2_l(pretrained=config['imgnet_pretrained'])
#         self.effnetv2l.features[0][0]=torch.nn.Conv2d(in_channels=1,out_channels=32, kernel_size=(3,3),stride=(2,2),padding=(1,1),bias=False)
#         # self.effnetv2l.classifier[1] = torch.nn.Identity()
#         self.num_ftrs_effnetv2l=self.effnetv2l.classifier[1].in_features 
        
#         self.effnetv2l.classifier[1] = torch.nn.Linear(self.num_ftrs_effnetv2l,1)
#         # self.effnetv2l=torch.nn.DataParallel(self.effnetv2l)
#         # self.eff_state_dict=torch.load('saved_models/effnetv2l_5_1')
#         # new_state_dict = {}
#         # for k, v in self.eff_state_dict.items():
#         #     k = k.replace("module.", "")
#         #     new_state_dict[k] = v
#         # state_dict = new_state_dict
        
#         # self.effnetv2l.load_state_dict(state_dict)
#         self.effnetv2l.classifier[1] = torch.nn.Linear(self.num_ftrs_effnetv2l,4096)
        
#         # SwinV2B part 
#         # Load the pretrained SwinV2-B model
#         self.swinv2b = models.swin_v2_b(pretrained=config['imgnet_pretrained'])

#         # Modify the first Conv2d layer to accept 1-channel input
#         self.swinv2b.features[0][0] = torch.nn.Conv2d(1, 128, kernel_size=(4, 4), stride=(4, 4))

#         # Replace the MLP block in each SwinTransformerBlockV2 with kanBlock
#         for name, module in self.swinv2b.named_modules():
#             if isinstance(module, type(self.swinv2b.features[1][0])):  # Checking for SwinTransformerBlockV2 type
#                 module.mlp = kanBlock(dim=module.mlp[0].in_features, num_heads=8)
        
        
#         # Adjust the final classification layer
#         self.num_ftrs_swinv2b = self.swinv2b.head.in_features
#         self.swinv2b.head = torch.nn.Linear(self.num_ftrs_swinv2b, 1)  # Single output for regression


#     def forward(self, x):
#         x1 = self.effnetv2l(x)
#         # print(x1.shape)
#         # x1=x1.detach()
#         x1=x1.reshape(-1,1,64,64)
#         # print(x1.shape)

#         x2 = self.swinv2b(x1)

#         return x2

    

