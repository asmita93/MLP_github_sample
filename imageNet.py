import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from imageNet_Datadings import ImagenetDatadings

# DATA
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225] 

class ImageNet():
    # num_workers = 0 in DDP , 8 otherwise
    def __init__(self, num_workers=0):
        self.batch_size = 512
        self.train_filename = '/ds/images/imagenet/msgpack/train.msgpack'
        self.val_filename = '/ds/images/imagenet/msgpack/val.msgpack'
        self.num_workers = num_workers
        self.get_transforms(224)
        self.setup()
        
    def get_transforms(self,input_size):

        self.train_transform = transforms.Compose([transforms.Resize(224),
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    
        Rs_size=int(input_size/0.9)

        self.val_transform = transforms.Compose(
        [transforms.Resize(Rs_size, interpolation=3),
         transforms.CenterCrop(input_size),
         transforms.ToTensor(),
         transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def setup(self):
        self.imagenet_train = ImagenetDatadings(self.train_filename, transform=self.train_transform)
        self.imagenet_val = ImagenetDatadings(self.val_filename, transform=self.val_transform)
  
    def train_dataloader(self):
        return DataLoader(self.imagenet_train, batch_size=self.batch_size, 
                          pin_memory=True, num_workers=self.num_workers)

  
    def test_dataloader(self):
         return DataLoader(self.imagenet_val, batch_size=self.batch_size,
                           pin_memory=True, num_workers=self.num_workers)
