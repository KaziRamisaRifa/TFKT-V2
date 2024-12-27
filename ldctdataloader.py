from torch.utils.data import Dataset, DataLoader
import torch
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from config import config
import tifffile
import numpy as np
import torchvision.transforms as T
from compimg.similarity import GSSIM
class Generate_data(Dataset):
    def __init__(self,length,data_type,cur_fold_image_list,all_labels,config):
        self.length=length
        self.data_type=data_type
        self.cur_fold_image_list=cur_fold_image_list
        self.all_labels=all_labels
        self.config=config.copy()
        self.transform_train = A.Compose(
           #Following list defines train augmentations
            [
                #A.RandomCrop(height=config['IM_W'],width=config['IM_H'],p=1,always_apply=True), 
                A.Resize(width=config['IM_W'], height=config['IM_H']),
                # A.RandomCrop(height=728, width=728),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
               #  A.Affine(
               #    scale=1.5,
               #    keep_ratio=True
               #  ),
               #  A.Rotate(limit=270),
                # A.Blur(p=0.8),
                # A.CLAHE(p=0.5),
                # A.ColorJitter(p=0.5),
                # A.CoarseDropout(max_holes=12, max_height=20, max_width=20, p=0.3),
                # A.IAAAffine(shear=30, rotate=0, p=0.2, mode="constant"),
                # A.Normalize(
                #     mean=[0.485, 0.456, 0.406],
                #     std=[0.229, 0.224, 0.225],    
                #     max_pixel_value=255.0
                # ),
                ToTensorV2(),
            ]
        )
        #another transformation for augmentation

        self.transform_val = A.Compose(
            [
               # Following list defines validation augmentations
            
                A.Resize(width=config['IM_W'], height=config['IM_H'],p=1),
                # A.Normalize(
                #     mean=[0.485, 0.456, 0.406],
                #     std=[0.229, 0.224, 0.225],    
                #     max_pixel_value=255.0,
                # ),
                ToTensorV2(),
            ]
        )
        self.transform_negative=A.Compose(
           [
              A.Resize(width=config['IM_W'], height=config['IM_H']),
              A.JpegCompression(p=.5),
              A.Blur(p=.5,blur_limit=4),
              A.RandomBrightness(p=0.5),
              A.GaussNoise(p=0.5,var_limit=1),
              A.GlassBlur(p=0.5),
              A.ColorJitter(p=0.5),
              ToTensorV2()

           ]
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.data_type=='train':
          if self.config['current_label']=='kadid':
             image=cv2.imread(self.cur_fold_image_list[idx])
             image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
          elif self.config['current_label']=='mayoct':
             original_image=tifffile.imread(config['mayoct_path']+self.cur_fold_image_list[idx])
          elif self.config['current_label']=='mayoct_classification':
             try:
                image=tifffile.imread(self.cur_fold_image_list[idx])
             except:
                print(self.cur_fold_image_list[idx])
          elif self.config['current_label']=='stdev':
             image=tifffile.imread(self.cur_fold_image_list[idx]) # Read image
             
          else:
             image=tifffile.imread('../LDCTIQAG2023_train/image/'+self.cur_fold_image_list[idx]) # Read image
          
          if config['current_label']=='mayoct':
             image=self.transform_negative(image=original_image)["image"]
          else:
             image=self.transform_train(image=image)["image"] #Perform augmentation


          if self.config['multi_channel_input']==True:
             # If multi channel input is true, then normalization is performed on the augmented image
             gaussed_image=T.GaussianBlur(kernel_size=(config['gauss_kernel'],config['gauss_kernel']))(image)
             gaussed_image=T.Resize(size=(config['IM_W']//2,config['IM_H']//2))(gaussed_image) # Downsampling by resize function by 1/2x
             gaussed_image=T.Resize(size=(config['IM_W'],config['IM_H']))(gaussed_image) # Upsampling by resize function by 2x

             lpf_normalized_image=image-gaussed_image # Subtracting the gaussian filtered image
             image=torch.cat((image,lpf_normalized_image),0) # Concatening the normalized image with original image
          if self.config['only_normalized']==True:
             # If single channel only normalized image
             gaussed_image=T.GaussianBlur(kernel_size=(config['gauss_kernel'],config['gauss_kernel']))(image)
             gaussed_image=T.Resize(size=(config['IM_W']//2,config['IM_H']//2))(gaussed_image)
             gaussed_image=T.Resize(size=(config['IM_W'],config['IM_H']))(gaussed_image)

             lpf_normalized_image=image-gaussed_image
             image=lpf_normalized_image
          
          image=(image-torch.min(image))/(torch.max(image)-torch.min(image)) # Performing min-max normalization
          if self.config['current_label']=='stdev':
             cur_img=image.flatten()
             temp_current_img=cur_img[cur_img >0.2]
             label=torch.std(temp_current_img)# If self-supervised stdev pretraining, then label will be the stdev of current image
            #  label=1.0-label # Since STDEV shold be opposite of IQA
          elif self.config['current_label']=='gt':
             label=self.all_labels[self.cur_fold_image_list[idx]] # if the current task is downstream, label will be the actual label
          elif self.config['current_label']=='kadid':
            label=self.all_labels[idx]
          elif self.config['current_label']=='mayoct_classification':
             label=self.all_labels[idx]
         #  print(label,self.cur_fold_image_list[idx])
          elif self.config['current_label']=='mayoct':
            #  np_org_img=original_image.data.cpu().numpy()
             np_distoted_img=image.data.cpu().numpy()
             np_distoted_img=np_distoted_img.squeeze()
             label=GSSIM().compare(original_image,np_distoted_img)

          
          return [image,label,self.cur_fold_image_list[idx]] # Retrun image, label, and image name
        
        if self.data_type=='val':
          if self.config['current_label']=='kadid':
             image=cv2.imread(self.cur_fold_image_list[idx])
             image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
          elif self.config['current_label']=='mayoct':
             original_image=tifffile.imread('../mayoct_full_reference_preprocessed_min_max/train/'+self.cur_fold_image_list[idx])
          else:
             image=tifffile.imread('../LDCTIQAG2023_train/image/'+self.cur_fold_image_list[idx]) # Read image
          if self.config['current_label']=='mayoct':
             image=self.transform_negative(image=original_image)["image"]
          else:
             
             image=self.transform_val(image=image)["image"] # Performing augmentation
          
          if self.config['multi_channel_input']==True:
             # In case of of multi-channel, in similar fashion, image normalization is performed here
             gaussed_image=T.GaussianBlur(kernel_size=(config['gauss_kernel'],config['gauss_kernel']))(image)
             gaussed_image=T.Resize(size=(config['IM_W']//2,config['IM_H']//2))(gaussed_image)
             gaussed_image=T.Resize(size=(config['IM_W'],config['IM_H']))(gaussed_image)
             lpf_normalized_image=image-gaussed_image
             image=torch.cat((image,lpf_normalized_image),0)

          if self.config['only_normalized']==True:
             # Similar to train, in case of only normalized input
             gaussed_image=T.GaussianBlur(kernel_size=(config['gauss_kernel'],config['gauss_kernel']))(image)
             gaussed_image=T.Resize(size=(config['IM_W']//2,config['IM_H']//2))(gaussed_image)
             gaussed_image=T.Resize(size=(config['IM_W'],config['IM_H']))(gaussed_image)

             lpf_normalized_image=image-gaussed_image
             image=lpf_normalized_image          
          image=(image-torch.min(image))/(torch.max(image)-torch.min(image)) # Min-max normalization
          if self.config['current_label']=='stdev':
             cur_img=image.flatten()
             temp_current_img=cur_img[cur_img >0.2]
             label=torch.std(temp_current_img) # Similar to train here as well, for self-supervised stdev pretraining label will be the stdev of current image
            #  label=1.0-label
          elif self.config['current_label']=='gt':
             label=self.all_labels[self.cur_fold_image_list[idx]] # if the current task is downstream, label will be the actual label
          elif self.config['current_label']=='kadid':
             label=self.all_labels[idx]
          elif self.config['current_label']=='mayoct':
            #  np_org_img=original_image.data.cpu().numpy()
             np_distoted_img=image.data.cpu().numpy()
             np_distoted_img=np_distoted_img.squeeze()

             label=GSSIM().compare(original_image,np_distoted_img)

          return [image,label,self.cur_fold_image_list[idx]]
        