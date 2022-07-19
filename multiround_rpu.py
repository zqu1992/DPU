import argparse

import torch
from torchvision import datasets, transforms 
import numpy as np   
import math
import time
import os


from model import MLP, VGG, ResNet56
from myoptimizer import PU_Adam_optimizer, PU_SGD_optimizer
from utils import train, train_ori, test, DPU_initialize, save_model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='./data/', help='Dataset directory')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Dataset, allowed: mnist, cifar10, cifar100')
    parser.add_argument('--model_dir', type=str, default='./model/', help='The directory to save the model')
    parser.add_argument('--model', type=str, default='vgg', help='The used inference model, allowed: mlp, vgg, resnet56')
    parser.add_argument('--mode', type=str, default='rpu', help='The partial updating mode (just for the saved model name), allowed: rpu')

    parser.add_argument('--val_dataset_size', type=int, default=3000, help='The number of samples in the validation dataset')
    parser.add_argument('--train_batch_size', type=int, default=128, help='The batch size used in training')
    parser.add_argument('--val_batch_size', type=int, default=128, help='The batch size used in validation')
    parser.add_argument('--test_batch_size', type=int, default=128, help='The batch size used in testing')
    parser.add_argument('--num_workers', type=int, default=1, help='The number of workers used in dataset preprocessing')
    parser.add_argument('--seed', type=int, default=1, help='The random seed')

    parser.add_argument('--updating_ratio', type=float, required=True, help='The partial updating ratio k, i.e., the ratio of updated weights')
    parser.add_argument('--lr', type=float, help='The initial learning rate')
    parser.add_argument('--num_epochs', type=int, help='The number of training epochs in each round')
    parser.add_argument('--reinit', action='store_true', help='Use re-initialization, default: do not use re-initialization')

    parser.add_argument('--dataset_size_init', type=int, required=True, help='The number of data samples in the initial dataset before deployment')
    parser.add_argument('--dataset_size_subseq', type=int, required=True, help='The number of data samples collected in each subsequent updating round')
    
    args = parser.parse_args()

    print(args)
    
    torch.backends.cudnn.benchmark = True

    # Set the random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Preprocess the dataset
    if args.dataset.lower() == 'mnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
           transforms.Normalize((0.1307,), (0.3081,)),
            ])
        train_dataset_full = datasets.MNIST(root=args.dataset_dir, train=True, download=True, transform=transform_train)
        test_dataset_full = datasets.MNIST(root=args.dataset_dir, train=False, download=True, transform=transform_test)
        num_classes = 10

    elif args.dataset.lower() == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        train_dataset_full = datasets.CIFAR10(root=args.dataset_dir, train=True, download=True, transform=transform_train)
        test_dataset_full = datasets.CIFAR10(root=args.dataset_dir, train=False, download=True, transform=transform_test)
        num_classes = 10

    elif args.dataset.lower() == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
            ])
        train_dataset_full = datasets.CIFAR100(root=args.dataset_dir, train=True, download=True, transform=transform_train)
        test_dataset_full = datasets.CIFAR100(root=args.dataset_dir, train=False, download=True, transform=transform_test)
        num_classes = 100

    else:
        print('The dataset: ', args.dataset, ' is not valid.')

    val_dataset, test_dataset = torch.utils.data.random_split(test_dataset_full, [args.val_dataset_size, len(test_dataset_full)-args.val_dataset_size])
    train_loader_full = torch.utils.data.DataLoader(train_dataset_full, batch_size=args.train_batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers) 
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)

    # Compute the number of updating rounds
    num_rounds = int(math.ceil((len(train_dataset_full)-args.dataset_size_init)/args.dataset_size_subseq)+1)

    print('The number of updating rounds: ', num_rounds)
    print('The updating ratio: ', args.updating_ratio)

    # Initialize a tensor to store the validation and test accuracy along the updating rounds
    round_accuracy = torch.zeros((num_rounds+1,2))

    # Initialize the random model 
    if args.model.lower() == 'mlp':
        model = MLP(num_classes=num_classes).cuda()
        args.num_epochs = args.num_epochs if args.num_epochs else 60
        args.lr = args.lr if args.lr else 5e-3
    elif args.model.lower() == 'vgg':
        model = VGG(num_classes=num_classes).cuda()
        args.num_epochs = args.num_epochs if args.num_epochs else 60
        args.lr = args.lr if args.lr else 5e-3
    elif args.model.lower() == 'resnet56':
        model = ResNet56(num_classes=num_classes).cuda()
        args.num_epochs = args.num_epochs if args.num_epochs else 100
        args.lr = args.lr if args.lr else 0.1
    else:
        print('The model: ', args.model, ' is not valid.')

    # Build the model path for saving
    init_model_path = args.model_dir + args.model + '_' + args.dataset + '_init.pth'
    current_model_path = args.model_dir + args.model + '_' + args.dataset + '_' + args.mode + '_current.pth'
    # Save the random initialized model
    # Comment out this instruction when the initial model is already available
    save_model(init_model_path, model, None, -1, -1)

    # Define the loss function
    loss_func = torch.nn.CrossEntropyLoss().cuda() 

    # Compute the validation and test accuracy of the random model
    _, val_accuracy = test(model, val_loader, loss_func, True)
    _, test_accuracy = test(model, test_loader, loss_func) 
    round_accuracy[0,0] = val_accuracy[0]
    round_accuracy[0,1] = test_accuracy[0]


    for r in range(num_rounds):

        print('Updating round: ', r)
        
        # Simulate the data collection process
        # Randomly drawn samples from the original training dataset without replacement
        if r == 0:
            train_dataset, train_dataset_subseq = torch.utils.data.random_split(train_dataset_full, [args.dataset_size_init, len(train_dataset_full)-args.dataset_size_init])
        else:
            if len(train_dataset_subseq) > args.dataset_size_subseq:
                train_dataset_new, train_dataset_subseq = torch.utils.data.random_split(train_dataset_subseq, [args.dataset_size_subseq, len(train_dataset_subseq)-args.dataset_size_subseq])
                train_dataset = torch.utils.data.ConcatDataset([train_dataset,train_dataset_new])
            else:
                train_dataset = torch.utils.data.ConcatDataset([train_dataset,train_dataset_subseq])
        # Build the training dataset loader
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
        
        print('The number of samples in the current training dataset: ', len(train_dataset))

        # Initialize optimizers for conv/fc weights and for other parameters
        if args.model.lower() == 'mlp':
            model = MLP(num_classes=num_classes).cuda()
            parameters_w = []
            parameters_b = []
            for name, param in model.named_parameters():
                if 'weight' in name and param.dim()>1: 
                    parameters_w.append(param)       
                else:
                    parameters_b.append(param)
            optimizer_b = torch.optim.Adam(parameters_b, lr=args.lr)
            scheduler_b = torch.optim.lr_scheduler.StepLR(optimizer_b, step_size=args.num_epochs//3, gamma=0.1)
            optimizer_w = PU_Adam_optimizer(parameters_w, lr=args.lr)
            scheduler_w = torch.optim.lr_scheduler.StepLR(optimizer_w, step_size=args.num_epochs//3, gamma=0.1)

        elif args.model.lower() == 'vgg':
            model = VGG(num_classes=num_classes).cuda()
            parameters_w = []
            parameters_b = []
            for name, param in model.named_parameters():
                if 'weight' in name and param.dim()>1: 
                    parameters_w.append(param)       
                else:
                    parameters_b.append(param)
            optimizer_b = torch.optim.Adam(parameters_b, lr=args.lr)
            scheduler_b = torch.optim.lr_scheduler.StepLR(optimizer_b, step_size=args.num_epochs//3, gamma=0.2)
            optimizer_w = PU_Adam_optimizer(parameters_w, lr=args.lr)
            scheduler_w = torch.optim.lr_scheduler.StepLR(optimizer_w, step_size=args.num_epochs//3, gamma=0.2)

        elif args.model.lower() == 'resnet56':
            model = ResNet56(num_classes=num_classes).cuda()
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

        
        if r == 0:
            print('Initialization...')
            # If the 0-th round, use the saved initialized random model
            num_samples = args.dataset_size_init
            last_init_round = 0
            checkpoint = torch.load(init_model_path)
        else:
            # Load the current deployed model, also the updated model from the last round
            checkpoint = torch.load(current_model_path)
            num_samples = checkpoint['num_samples']
            last_init_round = checkpoint['last_init_round']
            # Check if the number of newly collected data samples exceeds the number of samples when the model was re-initialized last time 
            if (len(train_dataset) > 2*num_samples) and args.reinit:
                print('Re-initialization...')
                # If yes, re-initialize the model by loading the saved initialized random model
                checkpoint = torch.load(init_model_path)
                # Update the relevant information for re-initialization
                num_samples = len(train_dataset)
                last_init_round = r 
            else:
                print('Load checkpoint of the last round...')
        model.load_state_dict(checkpoint['model_state_dict'])

        t = time.time()
        # Build the DPU module for each conv/fc layers
        DPU_layers = DPU_initialize(model, train_loader, loss_func) 
        # Record the validation and test accuracy at the start point in round r
        _, val_accuracy_init = test(model, val_loader, loss_func, True)
        _, test_accuracy_init = test(model, test_loader, loss_func)

        
        print('Randomly select updated weights...')
        # Randomly select updating_ratio of weights in each layer to update 
        for p_dpu in DPU_layers:
            perm = torch.randperm(p_dpu.w_init.nelement())
            updated_idx = perm[:round(p_dpu.w_init.nelement()*args.updating_ratio)]
            mask_new = torch.zeros(p_dpu.tensor_shape, dtype=torch.bool)
            mask_new.view(-1)[updated_idx] = 1
            p_dpu.replace_mask(mask_new)
        
        print('Currrent number of updated weights per layer: ')
        num_updated_weights_total = 0
        for p_dpu in DPU_layers:
            num_updated_weights_total += p_dpu.num_updated_weights
            print(p_dpu.num_updated_weights)
        print('Currrent total number of updated weights: ', num_updated_weights_total) 


        print('Sparsely fine-tuning...')
        # Conduct the sparse fine-tuning
        round_accuracy[r+1,0] = val_accuracy_init[0]
        round_accuracy[r+1,1] = test_accuracy_init[0]
        best_acc = val_accuracy_init[0]

        for epoch in range(args.num_epochs):
            scheduler_w.step(epoch=epoch)
            scheduler_b.step(epoch=epoch)
            train(model, train_loader, loss_func, optimizer_w, optimizer_b, DPU_layers, epoch, first_step=False)
            _, val_accuracy = test(model, val_loader, loss_func, True)
            # Check if the current model yields the highest validation accuracy  
            if val_accuracy[0] > best_acc:
                best_acc = val_accuracy[0]
                _, test_accuracy = test(model, test_loader, loss_func)
                save_model(current_model_path, model, DPU_layers, last_init_round, num_samples)
                round_accuracy[r+1,0] = val_accuracy[0]
                round_accuracy[r+1,1] = test_accuracy[0]
        print('In the '+str(r)+'-th round, partial updating costs: ', time.time()-t)



    print('The validation accuracy along rounds: ', round_accuracy[:,0])
    print('The test accuracy along rounds: ', round_accuracy[:,1])

    print('Avoid the accuracy degradation due to the re-initialization')
    # This kind of non-updating does not save communication, but improves the accuracy 
    for r in range(1,round_accuracy.size(0)):
        if round_accuracy[r,0].item() < round_accuracy[r-1,0].item():
            round_accuracy[r,1] = round_accuracy[r-1,1].item()
            round_accuracy[r,0] = round_accuracy[r-1,0].item()

    print('The validation accuracy along rounds: ', round_accuracy[:,0])
    print('The test accuracy along rounds: ', round_accuracy[:,1])

