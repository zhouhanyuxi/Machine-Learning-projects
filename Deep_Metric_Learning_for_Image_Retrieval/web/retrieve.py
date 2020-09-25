import pandas as pd
import torch
import PIL
from PIL import Image
import numpy as np
from torchvision import transforms
import sys

import contrastive
import triplet
import softtriple
import multisimilarity
import proxyNCA
import arcface
import cosface

modelpath = {"contrastive":"contrastive/contrastive.pth",
				"triplet":"triplet/triplet.pth",
				"softtriple":"softtriple/softtriple.pth",
				"cosface":"cosface/CosFace.pth",
				"multisimilarity":"multisimilarity/ms.pth",
				"arcface":"arcface/arcfacemodel.pth",
				"proxyNCA":"proxyNCA/ProxyNCA.pth",
				"0":"contrastive/contrastive.pth",
				"6":"triplet/triplet.pth",
				"1":"softtriple/softtriple.pth",
				"3":"cosface/CosFace.pth",
				"4":"multisimilarity/ms.pth",
				"2":"arcface/arcfacemodel.pth",
				"5":"proxyNCA/ProxyNCA.pth",
				"7":"softtriple/softtriple_cars.pth",
				"8":"multisimilarity/ms_car.pth",
				"9":"proxyNCA/ProxyNCA_cars.pth"}
				
csvpath = {"contrastive":"contrastive/contrastive.csv",
				"triplet":"triplet/triplet.csv",
				"softtriple":"softtriple/softtriple.csv",
				"cosface":"cosface/cosface1.csv",
				"multisimilarity":"multisimilarity/multisimilarity.csv",
				"arcface":"arcface/arcface.csv",
				"proxyNCA":"proxyNCA/ProxyNCA.csv",
				"0":"contrastive/contrastive.csv",
				"6":"triplet/triplet.csv",
				"1":"softtriple/softtriple.csv",
				"3":"cosface/cosface1.csv",
				"4":"multisimilarity/multisimilarity.csv",
				"2":"arcface/arcface.csv",
				"5":"proxyNCA/ProxyNCA.csv",
				"7":"softtriple/softtriple_cars.csv",
				"8":"multisimilarity/multisimilarity_cars.csv",
				"9":"proxyNCA/ProxyNCA_cars.csv"}

class retrieve():
    def __init__(self, model, model_path, embedding_path, image_path):
        # Rebuild model
        self.model = model
        self.model.load_state_dict(torch.load(model_path, map_location = torch.device('cpu')), strict = False)
        self.model.eval()
        # read embedding space
        self.train_embeddings_cl = pd.read_csv(embedding_path).values.tolist()
        # read image list
        self.image_list = pd.read_csv(image_path).values.tolist()

    def get_k_image_path(self, image, transform, k=10):
        #image = Image.open(new_image_path)
        image = transform(image)
        image.unsqueeze_(0)
        new_picture_embedding = self.model.forward(image).data.cpu().numpy()

        distance = []
        I = []
        for i in self.train_embeddings_cl:
            i = np.array(i)
            new_picture_embedding = np.array(new_picture_embedding)
            d = np.linalg.norm(new_picture_embedding - i)
            distance.append(d)

        for i in sorted(distance)[0:k]:
            I.append(distance.index(i)) 

        path_list = []
        for i in I:
            temp = self.image_list[i][0]
            temp = temp.replace("\\", '/')
            path_list.append(temp)
        return path_list

class Identity(): # used for skipping transforms
    def __call__(self, im):
        return im

class ScaleIntensities():
    def __init__(self, in_range, out_range):
        """ Scales intensities. For example [-1, 1] -> [0, 255]."""
        self.in_range = in_range
        self.out_range = out_range

    def __oldcall__(self, tensor):
        tensor.mul(255)
        return tensor

    def __call__(self, tensor):
        tensor = (
            tensor - self.in_range[0]
        ) / (
            self.in_range[1] - self.in_range[0]
        ) * (
            self.out_range[1] - self.out_range[0]
        ) + self.out_range[0]
        return tensor


class RGBToBGR():
    def __call__(self, im):
        assert im.mode == 'RGB'
        r, g, b = [im.getchannel(i) for i in range(3)]
        # RGB mode also for BGR, 3x8-bit pixels, true color, see PIL doc
        im = PIL.Image.merge('RGB', [b, g, r])
        return im

def make_transform(sz_resize = 256, sz_crop = 227, mean = [104, 117, 128],
        std = [1, 1, 1], rgb_to_bgr = True, is_train = True,
        intensity_scale = [[0, 1],[0, 255]]):
    return transforms.Compose([
        RGBToBGR() if rgb_to_bgr else Identity(),
        transforms.RandomResizedCrop(sz_crop) if is_train else Identity(),
        transforms.Resize((227,227)) if not is_train else Identity(),
        #transforms.CenterCrop(sz_crop) if not is_train else 
        Identity(),
        transforms.RandomHorizontalFlip() if is_train else Identity(),
        transforms.ToTensor(),
        ScaleIntensities(*intensity_scale) if intensity_scale is not None else Identity(),
        transforms.Normalize(
            mean=mean,
            std=std,
        )
    ])

def RGB2BGR(im):
    assert im.mode == 'RGB'
    r, g, b = [im.getchannel(i) for i in range(3)]
    return PIL.Image.merge('RGB', (b, g, r))

def MUL(x):
    return x.mul(255)

def trans(im):
    assert im.mode == 'RGB'
    r, g, b = [im.getchannel(i) for i in range(3)]
    # RGB mode also for BGR, 3x8-bit pixels, true color, see PIL doc
    im = PIL.Image.merge('RGB', [b, g, r])
    return im

def get_model_result(image, k=10, modelname="0", dataset = 'bird'):
    print(modelname)
    print(dataset)
    transform = None
    r = None
    path = sys.path[0] + '/static/'
    imagepath = ""
    if dataset == 'bird':
        imagepath = path + "image_path.csv"
    elif dataset == 'car':
        imagepath = path + "carsimage_path.csv"

    if modelname == "0" or modelname == "contrastive": #contrastive
        mean, std = 0.1307, 0.3081
        transform=transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                                                  transforms.Resize((28,28)),
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize((mean,), (std,))
                                                                  ])
        r = retrieve(model=contrastive.get_model(), model_path = path + "models/" + modelpath[modelname], embedding_path = path + "models/" + csvpath[modelname], image_path = imagepath)
    elif modelname == "6" or modelname == "triplet": #triplet
        mean, std = 0.1307, 0.3081
        transform=transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                                                  transforms.Resize((28,28)),
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize((mean,), (std,))
                                                                  ])
        r = retrieve(model=triplet.get_model(), model_path = path + "models/" + modelpath[modelname], embedding_path = path + "models/" + csvpath[modelname], image_path = imagepath)
    elif modelname == "3" or modelname == "cosface": # cosface
        transform = transforms.Compose([
            #transforms.RandomHorizontalFlip(),
            transforms.Resize((112,96)),
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
        ])
        r = retrieve(model=cosface.get_model(), model_path = path + "models/" + modelpath[modelname], embedding_path = path + "models/" + csvpath[modelname], image_path = imagepath)
    elif modelname == "1" or modelname == "7" or modelname == "softtriple": # softtriple
        model = softtriple.BNInception(dim=512)
        normalize = transforms.Normalize(mean=[104., 117., 128.],
                                 std=[1., 1., 1.])
        transform=transforms.Compose([
            transforms.Lambda(RGB2BGR),
            transforms.Resize((224,224)),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Lambda(MUL),
            normalize,
        ])
        r = retrieve(model=model, model_path = path + "models/" + modelpath[modelname], embedding_path = path + "models/" + csvpath[modelname], image_path = imagepath)
    elif modelname == "4" or modelname == "8" or modelname == "multisimilarity": # multisimilarity
        model = multisimilarity.BNInception(512)
        sz=224
        mean=[104, 117, 128]
        std=[1, 1, 1]
        transform=transforms.Compose([
            transforms.Lambda(trans),
            transforms.Resize((sz,sz)),
            transforms.ToTensor(),
            transforms.Lambda(MUL),
            transforms.Normalize(mean, std),
        ])
        r = retrieve(model=model, model_path = path + "models/" + modelpath[modelname], embedding_path = path + "models/" + csvpath[modelname], image_path = imagepath)
    elif modelname == "2" or modelname == "arcface":# arcface
        input_shape = [1, 128, 128]
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(input_shape[1:]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        r = retrieve(model=arcface.get_model(), model_path = path + "models/" + modelpath[modelname], embedding_path = path + "models/" + csvpath[modelname], image_path = imagepath)
    elif modelname == "5" or modelname == "9" or modelname == "proxyNCA": # proxyNCA
        transform = make_transform(is_train = False)
        r = retrieve(model=proxyNCA.get_model(), model_path = path + "models/" + modelpath[modelname], embedding_path = path + "models/" + csvpath[modelname], image_path = imagepath)

    result = r.get_k_image_path(image, transform, k)
    
    if dataset == 'bird':
        result = ["train/"+ r for r in result]
    else:
        result = ["cars/"+ r for r in result]

    return result
