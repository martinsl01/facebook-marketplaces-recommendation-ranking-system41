import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
import pandas as pd
from PIL import Image
from numpy import asarray
import torch

def  path_to_feature(path):
    image = Image.open(path)
    type(image)
    data = asarray(image)
    return data.reshape(-1)

class ProductsDataset(Dataset):

    def __init__(self, annotations_file = 'C:/Users/marti/FMRRS/training_data.csv', 
                 img_dir = 'C:/Users/marti/FMRRS/cleaned_images/', transform=None):
        super().__init__()
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) 

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, f"{self.img_labels.loc[index, 'image_id']}.jpg")
        image = path_to_feature(img_path)
        image_id = self.img_labels.loc[index, 'image_id']
        label = self.img_labels.loc[index, 'labels']
        image_id = self.transform(image)
    
        return torch.tensor(image_id), label


if __name__ == '__main__':
    dataset = ProductsDataset()
    train_dataloader = DataLoader(dataset, batch_size=16, shuffle=True)  
    train_images, train_labels = next(iter(train_dataloader))

print(len(dataset))
print(dataset[200])








