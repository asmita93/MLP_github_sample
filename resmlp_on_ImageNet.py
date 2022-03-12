import argparse
import os

from distributedDataParallelUtils import DistributedDataParellelUtils
from facade import Facade

def main(distributed, ngpus, mode,checkpoint_path):

    os.environ['WORLD_SIZE'] = ngpus

    # CUDA for PyTorch
    # self.use_cuda = torch.cuda.is_available()
    # self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
    # model = create_model().to(self.device)

    # These are the parameters used to initialize the process group
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK")
    }

    ddp = DistributedDataParellelUtils()
    f = Facade()
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    ddp.setup()
    device = int(os.environ['LOCAL_RANK'])

    print(
        f"[{os.getpid()}] world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()}"
        + f"device = local rank = {device}"
    )
    if(mode == "train"):
        f.train(distributed, ddp, device, checkpoint_path)
    if(mode == "test"):
        f.test(distributed, ddp, device, checkpoint_path)
    if(mode == "both"):
        f.train(distributed, ddp, device)
        f.test(distributed, ddp, device)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--distributed", type=bool, default=1, help='DistributedDataParellel is used or not')
    parser.add_argument("--ngpus", type=str, default='2', help="no. of tasks or World_size")
    parser.add_argument("--mode", type=str, help="train, test, both" )
    parser.add_argument("--checkpoint", type=str, help="load checkpoint")
    args = parser.parse_args()
    main(args.distributed, args.ngpus, args.mode, args.checkpoint)