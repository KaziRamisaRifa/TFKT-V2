import copy
import time
from collections import defaultdict
import tqdm
import torch
from utility import calc_loss, calc_other_metric, print_metrics
import json
import operator

def train_self_supervised(model, optimizer, scheduler, loss_fn, config, dataloader, fold):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    fold_performance = dict()

    for epoch in tqdm.tqdm(range(100)):
        for phase in ['train', 'val']:      
            if phase == 'train':               
                model.train()  
            else:
                model.eval()  
            metrics = defaultdict(float)
            epoch_samples = 0
            for btch, feed_dict in enumerate(dataloader[phase]):
                inputs = feed_dict[0]
                labels = feed_dict[1]

                inputs = inputs.to(config['device'])
                labels = labels.type(torch.FloatTensor)
                labels = labels.to(config['device'])

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)         
                    loss = calc_loss(outputs, labels, metrics, loss_fn, config, epoch + 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                epoch_samples += inputs.size(0) 

            epoch_loss = metrics['loss'] / epoch_samples

            if phase == 'val':
                if epoch_loss < best_loss:
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), 'saved_models/' + config['saved_model_name'] + '_' + str(fold))
                    fold_performance['self-supervised loss'] = epoch_loss

            if phase == 'train':
                scheduler.step()           
    print(fold_performance) 
    model.load_state_dict(best_model_wts)      
    return model

def train_model(model, optimizer, scheduler, loss_fn, config, dataloader, fold):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    best_krocc = -1e10
    fold_performance = dict()

    for epoch in tqdm.tqdm(range(config['epoch'])):
        since = time.time()
        for phase in ['train', 'val']:      
            if phase == 'train':               
                model.train()  
            else:
                model.eval()  
            metrics = defaultdict(float)
            epoch_samples = 0
            temp_epoch_outputs = []
            temp_epoch_labels = []
            org_epoch_labels = []
            temp_epoch_image_names = []
            for btch, feed_dict in enumerate(dataloader[phase]):
                inputs = feed_dict[0]
                labels = feed_dict[1]
                org_labels = labels

                img_names = feed_dict[2]

                inputs = inputs.to(config['device'])
                labels = labels.type(torch.FloatTensor)
                labels = labels.to(config['device'])

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)         
                    loss = calc_loss(outputs, labels, metrics, loss_fn, config=config, cur_epoch=epoch + 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                outputs = outputs.squeeze()
                labels = labels.squeeze()

                org_labels = org_labels.squeeze()

                outputs = outputs.data.cpu().numpy()
                labels = labels.data.cpu().numpy()
                org_labels = org_labels.data.cpu().numpy()

                outputs = outputs.flatten()
                labels = labels.flatten() 
                org_labels = org_labels.flatten()

                img_names = list(img_names)

                temp_epoch_outputs.extend(outputs)
                temp_epoch_labels.extend(labels)
                temp_epoch_image_names.extend(img_names)
                org_epoch_labels.extend(org_labels)

                epoch_samples += inputs.size(0) 

            calc_other_metric(temp_epoch_outputs, temp_epoch_labels, metrics, config=config)
            epoch_loss = metrics['loss'] / epoch_samples
            epoch_plcc = metrics['plcc'] / epoch_samples
            epoch_srocc = metrics['srocc'] / epoch_samples
            epoch_krocc = metrics['krocc'] / epoch_samples
            epoch_overall = metrics['overall'] / epoch_samples

            if phase == 'val':
                if config['all_data_training'] == True:
                    continue
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    torch.save(model.state_dict(), 'saved_models/' + config['saved_model_name'] + '_' + str(fold))

                    fold_performance['loss'] = epoch_loss
                    fold_performance['plcc'] = epoch_plcc
                    fold_performance['srocc'] = epoch_srocc
                    fold_performance['krocc'] = epoch_krocc
                    fold_performance['overall'] = epoch_overall

                    for label_idx in range(len(org_epoch_labels)):
                        org_epoch_labels[label_idx] = float(org_epoch_labels[label_idx])
                    for pred_idx in range(len(temp_epoch_outputs)):
                        temp_epoch_outputs[pred_idx] = float(temp_epoch_outputs[pred_idx])
                        if config['normalized_output'] == True:
                            temp_epoch_outputs[pred_idx] = temp_epoch_outputs[pred_idx] * 4.0

                    individual_outpus = []
                    for cur_epoch_img_name, cur_epoch_pred, cur_epoch_gt in zip(temp_epoch_image_names, temp_epoch_outputs, org_epoch_labels):
                        individual_outpus.append({
                            'img_name': cur_epoch_img_name,
                            'pred': cur_epoch_pred,
                            'gt': cur_epoch_gt
                        })
                    individual_outpus = sorted(individual_outpus, key=operator.itemgetter('gt'))

                    with open("individual_results/" + config['saved_model_name'] + '_' + str(fold) + '.json', "w") as individual_outfile:
                        json.dump(individual_outpus, individual_outfile)

            if phase == 'train':
                if config['all_data_training'] == True:
                    if epoch_loss < best_loss:
                        torch.save(model.state_dict(), 'saved_models/' + config['saved_model_name'] + '_' + str(fold))
                        best_loss = min(epoch_loss, best_loss)
                        fold_performance['loss'] = epoch_loss
                        fold_performance['plcc'] = epoch_plcc
                        fold_performance['srocc'] = epoch_srocc
                        fold_performance['krocc'] = epoch_krocc
                        fold_performance['overall'] = epoch_overall
                scheduler.step()           

    return fold_performance
