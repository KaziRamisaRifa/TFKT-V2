from scipy.stats import pearsonr, spearmanr, kendalltau
from torch.optim import lr_scheduler
import torch
import numpy as np
import statistics as stat
import os
import json 
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from torchmetrics.regression import KendallRankCorrCoef
from torch.nn import KLDivLoss
import torch

def get_loss_fn(config):
    if config['loss_fn']=='MSE':
        return torch.nn.MSELoss()
    if config['loss_fn']=='MAE':
        return torch.nn.L1Loss()
    if config['loss_fn']=='huber':
        return torch.nn.HuberLoss(delta=config['huber_delta'])

def get_scheduler(config,optimizer_ft):
    if config['scheduler']=='cos':
        scheduler_fn = lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=config['epoch']//config['scheduler_warmup'], eta_min=0,verbose=False)
    elif config['scheduler']=='step':
        scheduler_fn=lr_scheduler.StepLR(optimizer=optimizer_ft,step_size=config['scheduler_step'],verbose=False,gamma=0.1)
    return scheduler_fn

def calc_loss(preds,labels,metrics,loss_fn,config,cur_epoch,class_weights=None):
    preds=preds.squeeze()
    labels=labels.squeeze()

    if config['normalized_output']==True:
        labels=labels/4.0 # Incase, of normalized output labels are divided by 4.0

    loss=loss_fn(preds,labels)
    if config['add_KL']==True:
        # Since preds should be in log scale probability values.
        #REF: https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html 
        preds=torch.nn.functional.log_softmax(preds)
        labels=torch.nn.Softmax()(labels)
        loss=loss+config['KL_weight']*KLDivLoss()(preds,labels)

    if config['krocc_loss']==True:
        kendal = KendallRankCorrCoef()
        krocc_value=kendal(preds,labels)
        krocc_value=1-krocc_value
        loss=loss*krocc_value
    if config['rank_mse']==True:
        ranking_loss=torch.nn.BCELoss()
        target_vector=torch.zeros(labels.size(0))
        predicted_vector=torch.zeros(labels.size(0))
        for i in range(labels.size(0)):
            if i>0:
                if labels[i].item()>labels[i-1].item():
                    target_vector[i]=0
                    if preds[i].item()<preds[i-1].item():
                        predicted_vector[i]=1.0
                    else:
                        predicted_vector[i]=0
                if labels[i].item()<labels[i-1].item():
                    target_vector[i]=1.0
                    if preds[i].item()>preds[i-1].item():
                        predicted_vector[i]=0
                    else:
                        predicted_vector[i]=1.0
        
        loss=loss*ranking_loss(predicted_vector,target_vector)
        
        # mis_positioned=1
        # for i in range(labels.size(0)):
        #     if i>0:
        #         if labels[i].item()>labels[i-1].item():
        #             if preds[i].item()<preds[i-1].item():
        #                 mis_positioned=mis_positioned+1
        #         if labels[i].item()<labels[i-1].item():
        #             if preds[i].item()>preds[i-1].item():
        #                 mis_positioned=mis_positioned+1
        # loss=loss*mis_positioned
    if config['discordant_penalty']==True:
        mis_positioned=1
        for i in range(labels.size(0)):
            for j in range(i+1,labels.size(0)):
                if labels[i].item()>labels[j].item():
                    if preds[i].item()<preds[j].item():
                        mis_positioned=mis_positioned+1
                if labels[i].item()<labels[j].item():
                    if preds[i].item()>preds[j].item():
                        mis_positioned=mis_positioned+1
                if labels[i].item()==labels[j].item():
                    if preds[i].item()!=preds[j].item():
                        mis_positioned=mis_positioned+1
        loss=loss*mis_positioned

    
    
    if config['multi-task']==True:
        ce_loss=torch.nn.CrossEntropyLoss()
        classification_predictions=torch.zeros((labels.size(0),5))
        classification_labels=torch.zeros(labels.size(0))

        for i in range(preds.size(0)):
            if preds[i].item()<0.8:
                classification_predictions[i][0]=1.0
            elif 0.8<=preds[i].item() and preds[i].item()<1.6:
                classification_predictions[i][1]=1.0
            elif 1.6<=preds[i].item() and preds[i].item()<2.4:
                classification_predictions[i][2]=1.0
            elif 2.4<=preds[i].item() and preds[i].item()<3.2:
                classification_predictions[i][3]=1.0
            elif 3.2<=preds[i].item():
                classification_predictions[i][4]=1.0

        for i in range(labels.size(0)):
            if labels[i].item()<0.8:
                classification_labels[i]=0
            elif 0.8<=labels[i].item() and labels[i].item()<1.6:
                classification_labels[i]=1
            elif 1.6<=labels[i].item() and labels[i].item()<2.4:
                classification_labels[i]=2
            elif 2.4<=labels[i].item() and labels[i].item()<3.2:
                classification_labels[i]=3
            elif 3.2<=labels[i].item():
                classification_labels[i]=4
        classification_labels = classification_labels.type(torch.LongTensor)
        classification_labels = classification_labels.to(config['device'])
        classification_predictions=classification_predictions.to(config['device'])
        classification_loss=ce_loss(classification_predictions,classification_labels)
        weight_value=np.cos((cur_epoch/config['epoch'])*(np.pi/2))
        loss=((1-weight_value)*loss)+(weight_value*classification_loss)
    metrics['loss'] += loss.data.cpu().numpy() * labels.size(0)
    return loss


def calc_other_metric(preds,labels,metrics,config):

    # print(preds)
    # print(abs(pearsonr(labels, labels)[0]))
    preds=np.array(preds)
    if config['normalized_output']==True:
        preds=preds*4.0
    labels=np.array(labels)
    # print("inside calc other metric")
    # print(preds)
    # print(labels)
    # print(preds.shape)
    
    metrics["plcc"] += abs(pearsonr(preds, labels)[0]) * labels.shape[0]
    metrics["srocc"] += abs(spearmanr(preds, labels)[0]) * labels.shape[0]
    metrics["krocc"] += abs(kendalltau(preds, labels)[0]) * labels.shape[0]
    metrics["overall"] += (abs(pearsonr(preds, labels)[0]) + abs(spearmanr(preds, labels)[0]) + abs(kendalltau(preds, labels)[0])) * labels.shape[0]

def save_fold_results(wandb,all_fold_performance,config):
    columns = ["Fold","Loss","PLCC", "SROCC", "KROCC", "Overall"]
    loss_list,plcc_list,srocc_list,krocc_list,overall_list=[],[],[],[],[]
    
    for i in range(config['kfold']):
        loss_list.append(all_fold_performance[i]['loss'])
        plcc_list.append(all_fold_performance[i]['plcc'])
        srocc_list.append(all_fold_performance[i]['srocc'])
        krocc_list.append(all_fold_performance[i]['krocc'])
        overall_list.append(all_fold_performance[i]['overall'])
    data=[]
    for i in range(config['kfold']):
        data.append([str(i+1),all_fold_performance[i]['loss'],all_fold_performance[i]['plcc'],all_fold_performance[i]['srocc'],all_fold_performance[i]['krocc'],all_fold_performance[i]['overall']])
    if config['kfold']>1:
        data.append(['mean',stat.mean(loss_list),stat.mean(plcc_list),stat.mean(srocc_list),stat.mean(krocc_list),stat.mean(overall_list)])
        data.append(['std',stat.stdev(loss_list),stat.stdev(plcc_list),stat.stdev(srocc_list),stat.stdev(krocc_list),stat.stdev(overall_list)])

    # data = [['1',all_fold_performance[0]['loss'],all_fold_performance[0]['plcc'],all_fold_performance[0]['srocc'],all_fold_performance[0]['krocc'],all_fold_performance[0]['overall']],
    #         ['2',all_fold_performance[1]['loss'],all_fold_performance[1]['plcc'],all_fold_performance[1]['srocc'],all_fold_performance[1]['krocc'],all_fold_performance[1]['overall']],
    #         ['3',all_fold_performance[2]['loss'],all_fold_performance[2]['plcc'],all_fold_performance[2]['srocc'],all_fold_performance[2]['krocc'],all_fold_performance[2]['overall']],
    #         ['4',all_fold_performance[3]['loss'],all_fold_performance[3]['plcc'],all_fold_performance[3]['srocc'],all_fold_performance[3]['krocc'],all_fold_performance[3]['overall']],
    #         ['5',all_fold_performance[4]['loss'],all_fold_performance[4]['plcc'],all_fold_performance[4]['srocc'],all_fold_performance[4]['krocc'],all_fold_performance[4]['overall']],
    #         ['mean',stat.mean(loss_list),stat.mean(plcc_list),stat.mean(srocc_list),stat.mean(krocc_list),stat.mean(overall_list)],
    #         ['std',stat.stdev(loss_list),stat.stdev(plcc_list),stat.stdev(srocc_list),stat.stdev(krocc_list),stat.stdev(overall_list)]
    #         ]
    # print(data)
    if config['wandb']==True:
        val_table=wandb.Table(columns=columns,data=data)
        if config['kfold']<=1:
            wandb.log({"Test Table": val_table})
        else:
            wandb.log({"Val Table": val_table})
            

        for fold in range(1,config['kfold']+1):
            f = open('individual_results/'+config['saved_model_name']+'_'+str(fold)+'.json',"r")
            cur_fold_json = json.load(f)
            cur_fold_gt_list=[]
            cur_fold_pred_list=[]

            for cur_idx in range(len(cur_fold_json)):
                cur_fold_gt_list.append(cur_fold_json[cur_idx]['gt'])
                cur_fold_pred_list.append(cur_fold_json[cur_idx]['pred'])

                wandb.log({f"Fold {fold} img_name":cur_fold_json[cur_idx]['img_name'],
                        f"Fold {fold} predicted":cur_fold_json[cur_idx]['pred'],
                        f"Fold {fold} gt":cur_fold_json[cur_idx]['gt'], 
                        f"Fold {fold} idx":cur_idx,
                        f"Fold {fold} ABS error":abs(cur_fold_json[cur_idx]['gt']-cur_fold_json[cur_idx]['pred']),                     
                })
            cur_fold_r2_score=r2_score(cur_fold_gt_list,cur_fold_pred_list,force_finite=True)
            plt.scatter(cur_fold_gt_list,cur_fold_pred_list,color='green',s=2)
            plt.plot([min(cur_fold_gt_list), max(cur_fold_gt_list)], [min(cur_fold_pred_list), max(cur_fold_pred_list)], color='black', lw=2)
            plt.xlabel("Actual")
            plt.ylabel("Predicted")

            
            plt.title("R\u00b2:"+str(cur_fold_r2_score))
            plt.savefig('individual_experiment_plots/'+'r2_'+config['saved_model_name']+'_'+str(fold)+'_.png')
            plt.figure()

def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))

def fix_seed(config):
    config['saved_model_name']=config['model']+'_'+str(len(os.listdir('saved_models'))+1) # Unique (incremental) model name based on previous models in saved_models folders
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])
