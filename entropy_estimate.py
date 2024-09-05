from torchvision import datasets, transforms
import torch
from dataset import *
import pdb
import os
from torchvision.utils import save_image
from tqdm import tqdm
import argparse

rescaling     = lambda x : (x - .5) * 2.
rescaling_inv = lambda x : .5 * x  + .5
replicate_color_channel = lambda x : x.repeat(3,1,1)
average_color_channel = lambda x : torch.mean(x, dim=0, keepdim=True)

def set_data(dataset):
    print("Dataset: ", dataset)
    data_dir = '../data'
    batch_size = 1024
    apply_shuffle = True
    kwargs = {'num_workers':1, 'pin_memory':True, 'drop_last':False}
    
    if dataset == "mnist":
        ds_transforms = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), replicate_color_channel])
        train_loader = torch.utils.data.DataLoader(datasets.MNIST(data_dir, download=True, 
                            train=True, transform=ds_transforms), batch_size=batch_size, 
                                shuffle=apply_shuffle, **kwargs)
        
        test_loader  = torch.utils.data.DataLoader(datasets.MNIST(data_dir, train=False, 
                        transform=ds_transforms), batch_size=batch_size  , shuffle=False, **kwargs)
    elif dataset == "cifar10":
        ds_transforms = transforms.Compose([transforms.ToTensor()])
        train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(data_dir, train=True, 
            download=True, transform=ds_transforms), batch_size=batch_size, shuffle=apply_shuffle, **kwargs)
        
        test_loader  = torch.utils.data.DataLoader(datasets.CIFAR10(data_dir, train=False, 
                    transform=ds_transforms), batch_size=batch_size, shuffle=False, **kwargs)
    elif dataset == "cifar100":
        ds_transforms = transforms.Compose([transforms.ToTensor()])
        train_loader = torch.utils.data.DataLoader(datasets.CIFAR100(data_dir, train=True, 
            download=True, transform=ds_transforms), batch_size=batch_size, shuffle=apply_shuffle, **kwargs)
        
        test_loader  = torch.utils.data.DataLoader(datasets.CIFAR100(data_dir, train=False, 
                    transform=ds_transforms), batch_size=batch_size, shuffle=False, **kwargs)
    elif dataset == "cifar10_grey":
        ds_transforms = transforms.Compose([transforms.ToTensor(), average_color_channel, replicate_color_channel])
        train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(data_dir, train=True, 
            download=True, transform=ds_transforms), batch_size=batch_size, shuffle=apply_shuffle, **kwargs)
        
        test_loader  = torch.utils.data.DataLoader(datasets.CIFAR10(data_dir, train=False, 
                    transform=ds_transforms), batch_size=batch_size, shuffle=False, **kwargs)
    elif 'svhn' == dataset:
        ds_transforms = transforms.Compose([transforms.ToTensor()])
        train_loader = torch.utils.data.DataLoader(datasets.SVHN(data_dir, split='train', 
                download=True, transform=ds_transforms), batch_size=batch_size, shuffle=apply_shuffle, **kwargs)
        
        test_loader  = torch.utils.data.DataLoader(datasets.SVHN(data_dir, split='test',
                download=True, transform=ds_transforms), batch_size=batch_size, shuffle=False, **kwargs)
    elif 'celeba' == dataset:
        ds_transforms = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
        train_loader = torch.utils.data.DataLoader(datasets.CelebA(data_dir, split='train', 
                download=True, transform=ds_transforms), batch_size=batch_size, shuffle=apply_shuffle, **kwargs)
        
        test_loader  = torch.utils.data.DataLoader(datasets.CelebA(data_dir, split='test', 
                download=True, transform=ds_transforms), batch_size=batch_size, shuffle=False, **kwargs)
    elif 'fashionmnist' == dataset:
        ds_transforms = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), replicate_color_channel])
        train_loader = torch.utils.data.DataLoader(datasets.FashionMNIST(data_dir, download=True, 
                            train=True, transform=ds_transforms), batch_size=batch_size, 
                                shuffle=apply_shuffle, **kwargs)
        
        test_loader  = torch.utils.data.DataLoader(datasets.FashionMNIST(data_dir, train=False, download=True, 
                        transform=ds_transforms), batch_size=batch_size, shuffle=False, **kwargs)
    elif 'imagenet32' == dataset:
        ds_transforms = transforms.Compose([transforms.ToTensor()])
        train_loader = torch.utils.data.DataLoader(ImageNet32Dataset(os.path.join(data_dir, 'imagenet32'), train=True, 
                download=True, transform=ds_transforms), batch_size=batch_size, shuffle=apply_shuffle, **kwargs)
        test_loader = torch.utils.data.DataLoader(ImageNet32Dataset(os.path.join(data_dir, 'imagenet32'), train=False, 
                download=True, transform=ds_transforms), batch_size=batch_size, shuffle=apply_shuffle, **kwargs)
    elif 'sampling_cifar10' == dataset:
        ds_transforms = transforms.Compose([rescaling])
        train_loader = torch.utils.data.DataLoader(SamplingDataset(root_dir='./sampling_cifar10', 
                        transform=ds_transforms), batch_size=batch_size, shuffle=False, **kwargs)
        test_loader = None
    elif 'sampling_imagenet32' == dataset:
        ds_transforms = transforms.Compose([rescaling])
        train_loader = torch.utils.data.DataLoader(SamplingDataset(root_dir='./sampling_imagenet32', 
                        transform=ds_transforms), batch_size=batch_size, shuffle=False, **kwargs)
        test_loader = None
    else:
        raise Exception('{} dataset not in {mnist, cifar, sampling}'.format(dataset))
    
    return train_loader, test_loader

def save_img(tensor, save_dir, batch_idx):
    '''
    tensor should range from -1 to 1
    the shape of tensor should be (batch_size, 3, 32, 32)
    '''
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    tensor = (tensor + 1) / 2
    
    for i in range(tensor.shape[0]):
        save_image(tensor[i], os.path.join(save_dir, '{}.png'.format(batch_idx * tensor.shape[0] + i)))
    
# main func
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset to expand')
    args = parser.parse_args()
    
    train_loader, test_loader = set_data(args.dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args)
    print("Train: ", len(train_loader.dataset))
    print("Test: ", len(test_loader.dataset))
    count_tensor = torch.tensor(0).to(device)
    for batch_idx, (images, categories) in enumerate(tqdm(train_loader)):
        if images.max() > 1 or images.min() < 0:
            raise ValueError('Image range should be [0, 1]')
        else:
            images = (images * 255).to(torch.long).to(device)
            #onehot embedding
            one_hot_labels = torch.nn.functional.one_hot(images, num_classes=256)
            one_hot_sum = one_hot_labels.sum(dim=0)
            count_tensor = count_tensor + one_hot_sum
    
    count_tensor = count_tensor.float() + 1e-6        
    frequency = count_tensor / count_tensor.sum(dim = -1, keepdim = True)
    log_frequency = torch.log(frequency)
    entropy = - (frequency * log_frequency).sum(dim = -1).mean()
    # save frequency tensor
    torch.save(frequency, 'frequency_{}.pt'.format(args.dataset))
    print("dataset: ", args.dataset)
    print("Entropy: ", entropy.item())
    print("=" * 30)
