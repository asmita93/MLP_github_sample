import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm

from res_mlp_pytorch import ResMLP

class Resmlp():
    def __init__(self):
        
        # model
        self.model = ResMLP(
        image_size = 224,
        patch_size = 16,
        dim = 384,
        depth = 12,
        num_classes = 1000
        )

    def model_to_device_ddp(self, rank):
        self.model = self.model.to(rank)
        self.model = DDP(self.model, device_ids=[rank], output_device=rank)

    def optimizer(self):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.00088, weight_decay=0.067,
                                         betas=(0.0, 0.0),eps=1.0e-06)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 20, 1.0e-06)
    
    def train(self, train_data, device):
        
        epochs=600
        # n_samples=2500
        for epoch in range(epochs):

            train_data.sampler.set_epoch(epoch)
            correct = 0
            running_loss = 0.0
            total = 0
            with tqdm(train_data) as pbar:
                pbar.set_description(f"Epoch : {epoch +1}")
                for i,data in enumerate(tqdm(train_data)):

                    # if(i==n_samples):
                    #     break
                    
                    input, label = data
                    input = input.to(device)
                    label = label.to(device)

                    self.optimizer.zero_grad()

                    pred = self.model(input)

                    loss = self.criterion(pred, label)
                    loss.backward()
                    self.optimizer.step()

                     # print statistics
                    running_loss += loss.item()
                    _, predicted = torch.max(pred.data, 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()
                    pbar.update()
                    loss_epoch = running_loss/total
                    accuracy = 100*correct/total
                pbar.set_postfix_str('Train Loss: %.3f, Train Accuracy: %.3f'
                                         %(loss_epoch,accuracy))
            checkpoint = {'epoch': epoch, 'state_dict' : self.model.state_dict(), 'optimizer':self.optimizer.state_dict(), 'loss': loss_epoch, 
                            'accuracy': accuracy}

            if epoch%10 == 0:
                self.save_checkpoint(checkpoint, epoch)

            self.scheduler.step()
        print('Finished Training')

    def save_checkpoint(self, state, epoch):
        print("=> Saving checkpoint")
        filename = f"./resmlp_pth/checkpoint_{epoch}.pth.tar"
        torch.save(state, filename)

    def load_checkpoint(self,checkpoint):
        print("Loading checkpoint")
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        accuracy = checkpoint['accuracy']

    def save(self):
        
        self.PATH = './resmlp_pth/resmlp_imagenet_600epochs.pth'
        torch.save(self.model.state_dict(), self.PATH)
    
    def load(self):
        self.model = torch.load(self.PATH)
        # Print model's state_dict
        print("Model's state_dict:")
        for param_tensor in self.model.state_dict():
            print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())
        
    
    def test(self, test_data, rank):
        # Test on whole dataset
        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in test_data:
                images, labels = data
                input = input.to(rank)
                label = label.to(rank)
                # calculate outputs by running images through the network
                outputs = self.model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))

    def log(self):
        # Print model's state_dict
        print("Model's state_dict:")
        for param_tensor in self.model.state_dict():
            print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())

        # Print optimizer's state_dict
        print("Optimizer's state_dict:")
        for var_name in self.optimizer.state_dict():
            print(var_name, "\t", self.optimizer.state_dict()[var_name])
        
