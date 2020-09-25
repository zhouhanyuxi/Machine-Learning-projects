import torch.utils.data as data
from PIL import Image, ImageFile
import os
import cv2

ImageFile.LOAD_TRUNCATED_IAMGES = True


# https://github.com/pytorch/vision/issues/81

def PIL_loader(path):
    try:
        # print(path)
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')
    except IOError:
        print('Cannot load image ' + path)


def default_reader(fileList):
    imgList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            imgPath, label = line.strip().split(',')
            imgList.append(('/mnt/batch/tasks/shared/LS_root/mounts/clusters/yikai/code/Users/yili.lai/recognition/CUB_200_2011/images/' + imgPath, int(label)-1)) #local path needed
    return imgList


class ImageList(data.Dataset):
    def __init__(self, root, fileList, transform=None, list_reader=default_reader, loader=PIL_loader):
        self.root = root
        self.imgList = list_reader(fileList)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        imgPath, target = self.imgList[index]
        img = self.loader(os.path.join(self.root, imgPath))
        img = img.resize((112,96),Image.BILINEAR) # 缩放图片
        #img = cv2.imread(os.path.join(self.root, imgPath), cv2.IMREAD_COLOR)                        
        #img = cv2.resize(img, (250, 250), interpolation=cv2.INTER_LINEAR)

        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgList)
