import pandas as pd
import torch
from PIL import Image
import numpy as np

class retrieve():
    def __init__(self, model, model_path, embedding_path, image_path):
        # Rebuild model
        self.model = model
        self.model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')),strict=False)
        self.model.eval()
        # read embedding space
        self.train_embeddings_cl = pd.read_csv(embedding_path).values.tolist()
        # read image list
        self.image_list = pd.read_csv(image_path).values.tolist()

    def get_k_image_path(self,new_image_path, transform, k=10):
        image = Image.open(new_image_path)
        image = transform(image)
        image.unsqueeze_(0)
        new_picture_embedding = self.model.forward(image).data.cpu().numpy()

        distance = []
        I = []
        for i in self.train_embeddings_cl:
            i = np.array(i)
            new_picture_embedding = np.array(new_picture_embedding)
            d=np.linalg.norm(new_picture_embedding-i)
            distance.append(d)
            #d=0
            #for j in range(0,len(new_picture_embedding[0])):
            #    d += np.power(new_picture_embedding[0][j] - i[j], 2)
            #d = np.sqrt(d)
            #distance.append(d)

        for i in sorted(distance)[0:k]:
            I.append(distance.index(i)) 

        path_list = []
        for i in I:
            temp = self.image_list[i][0]
            temp = temp.replace("\\", '/')
            temp = '/mnt/batch/tasks/shared/LS_root/mounts/clusters/zhao2/code/Users/yikai.wang/ceshi/carsimages/' + temp
            #temp = '/mnt/batch/tasks/shared/LS_root/mounts/clusters/ton/code/Users/yikai.wang/ceshi/images/' + temp
            path_list.append(temp)
        return path_list