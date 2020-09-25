import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision import transforms


def get_dataloader(root_path, is_gray=None, size=None, batch_size=None, mean=None, std=None, transform=None):
    if transform is None:
        if is_gray:
            full_dataset = torchvision.datasets.ImageFolder(
            root=root_path,
            transform=transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                        transforms.Resize(size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((mean,), (std,))
                                        ]))
        else:
            full_dataset = torchvision.datasets.ImageFolder(
            root=root_path,
            transform=transforms.Compose([transforms.Resize(size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((mean,), (std,))
                                        ]))
        kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
        dataloader = torch.utils.data.DataLoader(full_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    else:
        full_dataset = torchvision.datasets.ImageFolder(
            root=root_path,
            transform=transform)
        kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
        dataloader = torch.utils.data.DataLoader(full_dataset, batch_size=batch_size, shuffle=False, **kwargs)
    return dataloader


def extract_embeddings(dataloader, model):
        with torch.no_grad():
            model.eval()
            embeddings = np.zeros((len(dataloader.dataset), 512))
            labels = np.zeros(len(dataloader.dataset))
            k = 0
            #print(len(dataloader[0]))
            #print(len(dataloader[1]))
            for images, target in dataloader:
                if torch.cuda.is_available():
                    images = images.cuda()
                embeddings[k:k+len(images)] = model.forward(images).data.cpu().numpy()
                labels[k:k+len(images)] = target.numpy()
                k += len(images)
        return embeddings, labels

'''
@:param model(class): 建立网络的类，需要包含get_model方法
@:param model_path(str):模型（.pth）的路径
@:param output_path(str):输出映射空间文件的路径（包含文件名）
@:param root_path(str):存放所有输入图片（用于映射空间）的文件夹目录。其中root_path路径下应为root_path/label1 ... root_path/labeln
@:param is_gray(boolean):预处理，图片是否处理为灰度， True代表是灰度，否则为RGB，默认为灰度
@:param size((int,int)):预处理，图片缩放的尺寸，默认为(28,28)
@:param batch_size(int):预处理，训练时模型使用的batch_size，默认为256
@:param mean(double):预处理，不知道干啥用的数，默认为0.1307
@:param std(double):预处理，不知道干啥用的数，默认为0.3081
@:param transform(torchvision.transforms):预处理，对于较为复杂的transform，将整个transform流程传入
'''

class extractor():

    def __init__(self, model, model_path, output_path, root_path, is_gray=True, size=(28,28), batch_size=256, mean=0.1, std=0.3, transform=None):
        self.model = model
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()
        self.output_path = output_path
        print('加载图片根目录：'+root_path)
        print('输出目录：'+output_path)
        if transform is None:
            print('是否灰度：'+str(is_gray))
            print('图片尺寸：'+str(size))
            print('batch_size：'+str(batch_size))
            self.dataloader = get_dataloader(
                                                root_path=root_path,
                                                is_gray=is_gray, 
                                                size=size, 
                                                batch_size=batch_size, 
                                                mean=mean, 
                                                std=std
                                            )
        else:
            self.dataloader = get_dataloader(
                                                root_path=root_path,
                                                transform=transform,
                                                batch_size=batch_size, 
                                            )

    
    def __getitem__(self):
        embeddings_cl, labels_cl = extract_embeddings(self.dataloader, self.model)
        embeddings_cl_df = pd.DataFrame(embeddings_cl)
        embeddings_cl_df.to_csv(self.output_path,index=False)
        print('Embedding Space Generated.')










        