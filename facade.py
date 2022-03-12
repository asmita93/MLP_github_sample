import torch
from resmlp import Resmlp
from imageNet import ImageNet


class Facade():

    def create_model(self):
        model = Resmlp()
        print("Model created")
        return model

    def get_data(self):
        data = ImageNet()
        print("Data acquired")
        return data

    def train(self, distributed, ddp, device, checkpoint_path):
        
        self.resmlp = self.create_model()
        self.resmlp.model_to_device_ddp(device)
    
        self.data = self.get_data() 
        
        self.resmlp.optimizer()

        if checkpoint_path:
            self.resmlp.load_checkpoint(torch.load(checkpoint_path))
        
        if distributed:
            train_set = self.data.imagenet_train
            train_loader = ddp.prepare(train_set, self.data.batch_size)
        else:
            train_loader = self.data.train_dataloader()

        self.resmlp.train(train_loader, device)   
        ddp.cleanup()
        self.resmlp.save()
        # model.log()

        print("Training complete.")

    def test(self, distributed, ddp, device, checkpoint_path=None):

        self.resmlp = self.create_model()
        self.resmlp.model_to_device_ddp(device)
            
        self.data = self.get_data() 
        
        if checkpoint_path:
            self.resmlp.load_checkpoint(torch.load(checkpoint_path)) #"./cifar_mixer_pth/checkpoint_5.pth"

        # self.resmlp.load()
        
        if distributed:
            test_set = self.data.imagenet_val
            test_loader = ddp.prepare(test_set, self.data.batch_size)
        else:
            test_loader = self.data.test_dataloader()

        self.resmlp.test(test_loader, device)   
        ddp.cleanup()