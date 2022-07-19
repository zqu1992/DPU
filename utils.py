import torch
import numpy as np   
import time

from module import DPU_module

TOPK = (1,5)

def accuracy(output, target, correct_sum, topk=(1,)):
    """Compute the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        for (i,k) in enumerate(topk):
            correct_sum[i] += (correct[:k].reshape(-1).float().sum(0, keepdim=True)).item()
        return 


def train_ori(model, train_loader, loss_func, optimizer, epoch):
    """Train the model with original optimizer"""
    model.train()
    epoch_train_loss = 0.0
    correct_sum = [0. for i in range(len(TOPK))]
    num_samples = 0
    for inputs, labels in train_loader:               
        inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        optimizer.zero_grad()    
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()   
        optimizer.step()
        accuracy(outputs, labels, correct_sum, topk=TOPK)
        num_samples += labels.size(0)
        epoch_train_loss += loss.item()
    print("epoch: ", epoch, ", training loss: ", epoch_train_loss/len(train_loader))  
    print('training accuracy: ', [ci/num_samples for ci in correct_sum])


def train(model, train_loader, loss_func, optimizer_w, optimizer_b, DPU_layers, epoch, first_step=False):
    """Train the model with DPU optimizer"""
    model.train()
    epoch_train_loss = 0.0
    correct_sum = [0. for i in range(len(TOPK))]
    num_samples = 0
    for inputs, labels in train_loader:               
        inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        optimizer_w.zero_grad()
        optimizer_b.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()   
        optimizer_b.step()
        optimizer_w.step(DPU_layers, first_step)
        accuracy(outputs, labels, correct_sum, topk=TOPK)
        num_samples += labels.size(0)
        epoch_train_loss += loss.item()
    print("epoch: ", epoch, ", loss: ", epoch_train_loss/len(train_loader))  
    print('training accuracy: ', [ci/num_samples for ci in correct_sum])
   

def test(model, test_loader, loss_func, val=False):
    """Validate or test the model"""
    model.eval()
    correct_sum = [0. for i in range(len(TOPK))]
    num_samples = 0
    current_test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            outputs = model(inputs)
            loss = loss_func(outputs, labels)  
            accuracy(outputs, labels, correct_sum, topk=TOPK)
            num_samples += labels.size(0)
            current_test_loss += loss.item()
        test_loss = current_test_loss/len(test_loader)
        test_accuracy = [ci/num_samples for ci in correct_sum]
        if val:
            # print("validation loss: ", test_loss)
            print("validation accuracy: ", test_accuracy)
        else:
            # print("test loss: ", test_loss)
            print("test accuracy: ", test_accuracy)
        return test_loss, test_accuracy
        

def DPU_initialize(model, train_loader, loss_func):
    """Initialize the DPU modules"""
    layer_idx = 0
    DPU_layers = []
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim()>1:
            DPU_layers.append(DPU_module(param, layer_idx))  
            layer_idx += 1        
    train_loss, train_accuracy = test(model, train_loader, loss_func)
    print("initial training loss: ", train_loss)  
    print('initial training accuracy: ', train_accuracy)
    print('currrent number of updated weights per layer: ')
    for p_dpu in DPU_layers:
        print(p_dpu.num_updated_weights)
    return DPU_layers 

  

def save_model(file_name, model, DPU_layers, last_init_round, num_samples):
    """Save the model, DPU modules, the round number of the last re-initialization, and the number of samples at the last re-initialization"""
    print('saving...')   
    torch.save({
        'model_state_dict': model.state_dict(),
        'DPU_layers': DPU_layers,
        'last_init_round': last_init_round,
        'num_samples': num_samples,
        }, file_name)

