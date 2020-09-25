import torch
import torchvision.transforms as T

class Config:
    # network settings
    backbone = 'resnet' # [resnet, fmobile]
    metric = 'arcface'  # [cosface, arcface]
    embedding_size = 512
    drop_ratio = 0.5

    # data preprocess
    input_shape = [1, 128, 128]
    train_transform = T.Compose([
        T.Grayscale(),
        T.RandomHorizontalFlip(),
        T.Resize((144, 144)),
        T.RandomCrop(input_shape[1:]),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])
    test_transform = T.Compose([
        T.Grayscale(),
        T.Resize(input_shape[1:]),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])

    # dataset
    train_root = 'CUB_200_2011/train'
    test_root = "lfw-align-128"
    test_list = "lfw_test_pair.txt"
    
    # training settings
    checkpoints = "checkpoints"
    restore = False
    restore_model = "arcfacemodel.pth"
    test_model = "checkpoints/24.pth"
    
    train_batch_size = 64
    test_batch_size = 60

    epoch = 20
    optimizer = 'adam'  # ['sgd', 'adam']
    lr = 5e-4
    lr_step = 10
    lr_decay = 0.95
    weight_decay = 5e-4
    loss = 'focal_loss' # ['focal_loss', 'cross_entropy']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pin_memory = True  # if memory is large, set it True to speed up a bit
    num_workers = 0  # dataloader

config = Config()
