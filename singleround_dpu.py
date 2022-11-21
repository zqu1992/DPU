import argparse

import torch
from torchvision import datasets, transforms 
import numpy as np   
import math
import time
import os


from model import MobileNetV1
from myoptimizer import PU_Adam_optimizer, PU_SGD_optimizer
from utils import train, train_ori, test, DPU_initialize, save_model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='./data/', help='Dataset directory')
    parser.add_argument('--dataset', type=str, default='imagenet', help='Dataset, allowed: imagenet')
    parser.add_argument('--model_dir', type=str, default='./model/', help='The directory to save the model')
    parser.add_argument('--model', type=str, default='mobilenetv1', help='The used inference model, allowed: mobilenetv1')
    parser.add_argument('--mode', type=str, default='single_dpu', help='The partial updating mode (just for the saved model name), allowed: single_dpu')

    parser.add_argument('--val_dataset_size', type=int, default=15000, help='The number of samples in the validation dataset')
    parser.add_argument('--train_batch_size', type=int, default=1024, help='The batch size used in training')
    parser.add_argument('--val_batch_size', type=int, default=1024, help='The batch size used in validation')
    parser.add_argument('--test_batch_size', type=int, default=1024, help='The batch size used in testing')
    parser.add_argument('--num_workers', type=int, default=32, help='The number of workers used in dataset preprocessing')
    parser.add_argument('--seed', type=int, default=1, help='The random seed')

    parser.add_argument('--updating_ratio', type=float, required=True, help='The partial updating ratio k, i.e., the ratio of updated weights')
    parser.add_argument('--lr', type=float, help='The initial learning rate')
    parser.add_argument('--num_epochs', type=int, help='The number of training epochs')
    parser.add_argument('--iterative_fixing_ratio', type=float, default=0.2, help='The ratio , allowed: [0,1]')
    parser.add_argument('--lamda', type=float, default=0.5, help='The ratio when adding both normalized contributions, allowed: [0,1]')

    parser.add_argument('--dataset_size_init', type=int, default=8e5, help='The number of data samples in the initial dataset before deployment')
    parser.add_argument('--dataset_size_subseq', type=int, default=4.8e5, help='The number of data samples collected in the subsequent updating round')
    
    args = parser.parse_args()

    print(args)
    
    torch.backends.cudnn.benchmark = True

    # Set the random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Preprocess the dataset
    if args.dataset.lower() == 'imagenet':
        train_dir = os.path.join(args.dataset_dir, 'train')
        val_dir = os.path.join(args.dataset_dir, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        train_dataset_full = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        val_dataset_full = datasets.ImageFolder(
            valdir, 
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
        num_classes = 1000

    else:
        print('The dataset: ', args.dataset, ' is not valid.')

    val_dataset, test_dataset = torch.utils.data.random_split(val_dataset_full, [args.val_dataset_size, len(val_dataset_full)-args.val_dataset_size])
    train_loader_full = torch.utils.data.DataLoader(train_dataset_full, batch_size=args.train_batch_size, shuffle=True, pin_memory=False, num_workers=args.num_workers) 
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=True, pin_memory=False, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True, pin_memory=False, num_workers=args.num_workers)


    print('The number of updating rounds: ', 1)
    print('The updating ratio: ', args.updating_ratio)

    # Initialize a tensor to store the validation and test accuracy along the updating rounds
    round_accuracy = torch.zeros((3,2))

    # Initialize the random model 
    if args.model.lower() == 'mobilenetv1':
        model = MobileNetV1(num_classes=num_classes)
        model = torch.nn.DataParallel(model).cuda()
        args.num_epochs = args.num_epochs if args.num_epochs else 150
        args.lr = args.lr if args.lr else 0.5
    else:
        print('The model: ', args.model, ' is not valid.')

    # Build the model path for saving
    init_model_path = args.model_dir + args.model + '_' + args.dataset + '_init.pth'
    current_model_path = args.model_dir + args.model + '_' + args.dataset + '_' + args.mode + '_current.pth'
    # Save the random initialized model
    # Comment out this instruction when the initial model is already available
    save_model(init_model_path, model.module, None, -1, -1)

    # Define the loss function
    loss_func = torch.nn.CrossEntropyLoss().cuda() 

    # Compute the validation and test accuracy of the random model
    _, val_accuracy = test(model, val_loader, loss_func, True)
    _, test_accuracy = test(model, test_loader, loss_func) 
    round_accuracy[0,0] = val_accuracy[0]
    round_accuracy[0,1] = test_accuracy[0]


    # Simulate the data collection process
    # Randomly drawn samples from the original training dataset without replacement
    train_dataset_init, train_dataset_subseq = torch.utils.data.random_split(train_dataset_full, [args.dataset_size_init, len(train_dataset_full)-args.dataset_size_init])
    if len(train_dataset_subseq) > args.dataset_size_subseq:
        train_dataset_new, _ = torch.utils.data.random_split(train_dataset_subseq, [args.dataset_size_subseq, len(train_dataset_subseq)-args.dataset_size_subseq])
        train_dataset_final = torch.utils.data.ConcatDataset([train_dataset_init,train_dataset_new])
    else:
        print('There are not enough data samples in the original training dataset.')
        print('The number of newly collected data samples after initial deployment is: ', len(train_dataset_subseq))
        train_dataset_final = train_dataset_full
    
    # Build the training dataset loader
    train_loader_init = torch.utils.data.DataLoader(train_dataset_init, batch_size=args.train_batch_size, shuffle=True, pin_memory=False, num_workers=args.num_workers)
    train_loader_final = torch.utils.data.DataLoader(train_dataset_final, batch_size=args.train_batch_size, shuffle=True, pin_memory=False, num_workers=args.num_workers)
    
    print('The number of samples in the initial training dataset: ', len(train_dataset_init))
    print('The number of samples in the final training dataset: ', len(train_dataset_final))

    print('Initialization...')
    num_samples = args.dataset_size_init
    last_init_round = 0
    # Use the saved initialized random model (can be omitted if not using the saved random model)
    checkpoint = torch.load(init_model_path)
    model.module.load_state_dict(checkpoint['model_state_dict'])


    print('Training (full updating) for the initial deployment...')
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=0)
    # Conduct the full updating on the initial training dataset for the initial deployment 
    best_acc = 0.
    for epoch in range(args.num_epochs):
        scheduler.step(epoch=epoch)
        train_ori(model, train_loader_init, loss_func, optimizer, epoch)
        _, val_accuracy = test(model, val_loader, loss_func, True)
        # Check if the current model yields the highest validation accuracy  
        if val_accuracy[0] > best_acc:
            best_acc = val_accuracy[0]
            _, test_accuracy = test(model, test_loader, loss_func)
            save_model(current_model_path, model.module, None, last_init_round, num_samples)
            round_accuracy[1,0] = val_accuracy[0]
            round_accuracy[1,1] = test_accuracy[0]


    print('Partial updating the initially deployed model in a single round...')
    model = MobileNetV1(num_classes=num_classes)
    model = torch.nn.DataParallel(model).cuda()
    # Initialize optimizers for conv/fc weights and for other parameters
    parameters_w = []
    parameters_b = []
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim()>1: 
            parameters_w.append(param)       
        else:
            parameters_b.append(param)
    optimizer_b = torch.optim.SGD(parameters_b, lr=args.lr, momentum=0.9, weight_decay=0.0001, nesterov=True)
    scheduler_b = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_b, T_max=args.num_epochs, eta_min=0)
    optimizer_w = PU_SGD_optimizer(parameters_w, lr=args.lr, momentum=0.9, weight_decay=0.0001, nesterov=True)
    scheduler_w = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_w, T_max=args.num_epochs, eta_min=0)

    # Load the current deployed model, also the initially deployed model
    # No re-initialization here
    print('Load checkpoint of the initially deployed model...')
    checkpoint = torch.load(current_model_path)
    num_samples = checkpoint['num_samples']
    last_init_round = checkpoint['last_init_round']
    model.module.load_state_dict(checkpoint['model_state_dict'])
    
    # Build the DPU module for each conv/fc layers
    # For the sparse initially deployed model, the member variable "mask" in DPU_module must be set manually, where nonzero weights have mask value 1
    DPU_layers = DPU_initialize(model, train_loader_final, loss_func) 
    # Record the validation and test accuracy at the start point of the partial updating, also the initially deployed model
    _, val_accuracy_init = test(model, val_loader, loss_func, True)
    _, test_accuracy_init = test(model, test_loader, loss_func)

    print('Full updating...')   
    # Conduct the first step, also a full updating
    for epoch in range(args.num_epochs):
        scheduler_w.step(epoch=epoch)
        scheduler_b.step(epoch=epoch)
        train(model, train_loader_final, loss_func, optimizer_w, optimizer_b, DPU_layers, epoch, first_step=True)

    num_weights_total = 0
    num_updated_weights_total = 0
    for p_dpu in DPU_layers:
        num_updated_weights_total += p_dpu.num_updated_weights
        num_weights_total += p_dpu.w_init.nelement()
    current_updating_ratio = num_updated_weights_total/num_weights_total
    print('Currrent ratio of updated weights: ', current_updating_ratio)

    while current_updating_ratio > args.updating_ratio:
        # Set the updating ratio of the current rewinding iteration
        # Check if reaching the target the updating ratio
        if current_updating_ratio*(1-args.iterative_fixing_ratio) > args.updating_ratio:
            current_updating_ratio = current_updating_ratio*(1-args.iterative_fixing_ratio)
        else:
            current_updating_ratio = args.updating_ratio


        print('Rewinding...')
        sum_global_contribution = 0.
        sum_local_contribution = 0.
        for p_dpu, p in zip(DPU_layers, parameters_w):
            # Compute the global contribution
            p_dpu.global_contribution.add_(torch.pow(p.data-p_dpu.w_init,2))
            # Compute the sum of all global contribution across all layers
            sum_global_contribution += torch.sum(p_dpu.global_contribution).item()
            # Compute the sum of all local contribution across all layers
            sum_local_contribution += torch.sum(p_dpu.local_contribution).item()
        # Normalize the global and local contributions
        for p_dpu in DPU_layers:
            p_dpu.global_contribution.div_(sum_global_contribution)
            p_dpu.local_contribution.div_(sum_local_contribution)
        # Compute the rewinding metric of the combined contribution
        for p_dpu in DPU_layers:
            p_dpu.metric.add_(args.lamda*p_dpu.global_contribution+(1-args.lamda)*p_dpu.local_contribution)
        
        # Sort the metric across all layers and rewind the weights to their initial values according to the updating ratio in the current iteration
        metric_list = torch.tensor([]).cuda()
        num_weights_total = 0
        for i, (p_dpu, p) in enumerate(zip(DPU_layers, parameters_w)):
            # Choose the top-65% weights in the i-th layer for the later global sorting
            # Sometimes, with a limited memory, an entire global sorting may be not possible, e.g., you may use 0.65 here when training VGG 
            i_metric_list = p_dpu.sort_metric(int(p_dpu.w_init.nelement()*1.))    
            metric_list = torch.cat((metric_list, i_metric_list), 0)
            num_weights_total += p_dpu.w_init.nelement()
        # Conduct global sorting
        sorted_idx = torch.argsort(metric_list[:,-1], descending=True)
        # Find the top-k*I weights according to their metric values
        act_idx = sorted_idx[:int(current_updating_ratio*num_weights_total)]
        act_list = metric_list[act_idx,:]
        # Rewind the weights and generate the start point of the second step
        for i, (p_dpu, p) in enumerate(zip(DPU_layers, parameters_w)):
            p_dpu.update_mask(act_list[act_list[:,0]==i,1].long())
            p.data.mul_(p_dpu.mask).add_(torch.mul(p_dpu.w_init,~p_dpu.mask))

        # Reset global and local contributions for the next iteration
        for p_dpu in DPU_layers:
            p_dpu.global_contribution.zero_()
            p_dpu.metric.zero_()
            p_dpu.local_contribution.mul_(sum_local_contribution) 
            p_dpu.local_contribution.mul_(p_dpu.mask)

        # Record the validation and test accuracy at the start point of the second step 
        _, val_accuracy_rewind = test(model, val_loader, loss_func, True)
        _, test_accuracy_rewind = test(model, test_loader, loss_func)
        
        print('Currrent number of updated weights per layer: ')
        num_updated_weights_total = 0
        for p_dpu in DPU_layers:
            num_updated_weights_total += p_dpu.num_updated_weights
            print(p_dpu.num_updated_weights)
        print('Currrent total number of updated weights: ', num_updated_weights_total) 
        current_updating_ratio = num_updated_weights_total/num_weights_total
        print('Currrent ratio of updated weights: ', current_updating_ratio)

        print('Sparsely fine-tuning...')
        # Conduct the second step, also the first step for the next iteration
        save_model(current_model_path, model.module, DPU_layers, last_init_round, num_samples)
        round_accuracy[2,0] = val_accuracy_rewind[0]
        round_accuracy[2,1] = test_accuracy_rewind[0]
        best_acc = val_accuracy_rewind[0]
        for epoch in range(args.num_epochs):
            scheduler_w.step(epoch=epoch)
            scheduler_b.step(epoch=epoch)
            train(model, train_loader_final, loss_func, optimizer_w, optimizer_b, DPU_layers, epoch, first_step=True)
            _, val_accuracy = test(model, val_loader, loss_func, True)
            # Check if the current model yields the highest validation accuracy  
            if val_accuracy[0] > best_acc:
                best_acc = val_accuracy[0]
                _, test_accuracy = test(model, test_loader, loss_func)
                save_model(current_model_path, model.module, DPU_layers, last_init_round, num_samples)
                round_accuracy[2,0] = val_accuracy[0]
                round_accuracy[2,1] = test_accuracy[0]


    print('The validation accuracy along rounds: ', round_accuracy[:,0])
    print('The test accuracy along rounds: ', round_accuracy[:,1])