import argparse
import torch
import torch.distributed as dist
from torchvision.models import resnet18
from torch.nn.parallel import DistributedDataParallel
import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
import os

# notice the `local_rank` will be automatic setting by launch script

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
# --nnodes 1 --node_rank 0 --nproc_per_node 4 \
# --master_addr 127.0.0.1 --master_port 29500 \
# dist_example.py

parser = argparse.ArgumentParser()
# https://docs.pytorch.org/docs/stable/distributed.html#launch-utility
# pytorch-2.0 prefer `--local-rank` rather than `--local_rank`
parser.add_argument("--local-rank", type=int, default=0, required=True)
args = parser.parse_args()

# ============ environment variable from `torch/distributed/launch.py` script =============
# rank = os.environ["RANK"]
# master_addr = os.environ["MASTER_ADDR"]
# master_port = os.environ["MASTER_PORT"]
world_size = int(os.environ["WORLD_SIZE"])
# local_rank = int(os.environ["LOCAL_RANK"])  # This also works
local_rank = args.local_rank

# =========== initialize configures ===================
# `init_method` has default value `"env://"`
dist.init_process_group(backend='nccl', init_method="env://")
torch.cuda.set_device(local_rank)
torch.backends.cudnn.benchmark = True  # Let torch to choose suitable conv algorithm automatic
use_amp = True
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# =========== define network, optimizer, loss, scheduler ===========
model = resnet18()
# model.cuda()  # This also works
model.cuda(local_rank)
model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)
loss_fn = nn.CrossEntropyLoss().cuda(local_rank)
# scheduler = torch.optim.LambdaLR(optimizer, lambda epoch: 1 / (epoch + 1)**2)  # optional

# ========== prepare dataset and dataloader ==========
class ExampleDataset(torch.utils.data.Dataset):
    def __init__(self, samples=128):
        self.samples = samples
        self.num_classes = 1000
        self.x = torch.rand((samples, 3, 224, 224))
        self.y = torch.randint(0, self.num_classes, size=(samples,))
    def __len__(self):
        return self.samples
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

train_dataset = ExampleDataset(256)
test_dataset = ExampleDataset(128)
train_sampler = DistributedSampler(train_dataset, shuffle=True)
test_sampler = DistributedSampler(test_dataset, shuffle=False)

# ================ some help function for evaluation ===============
def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

# ============ main loop ================
for epoch in range(10):
    train_sampler.set_epoch(epoch)
    test_sampler.set_epoch(epoch)
    # correct: When use DistributedSampler, the `batch_size` here is batch_size_per_gpu
    train_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, batch_size=64, num_workers=2)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, sampler=test_sampler, batch_size=64, num_workers=2)

    # ===================== train loop ===========================
    model.train()
    for batch_x, batch_y in train_dataloader:
        batch_x = batch_x.cuda(local_rank, non_blocking=True)
        batch_y = batch_y.cuda(local_rank, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            output = model(batch_x)
            loss = loss_fn(output, batch_y)
        scaler.scale(loss).backward()
        # if use clip gradient norm, uncomment the following two lines
        # scaler.unscale_(optimizer)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    # scheduler.step()  # optional

    # ===================== evaluate loop ===================
    # model.eval has no relation with torch.no_grad()
    model.eval()
    # Warning: without this line, a new graph well be created, thus the GPU memory may be 2x
    with torch.no_grad():
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        for batch_x, batch_y in test_dataloader:
            batch_x = batch_x.cuda(local_rank, non_blocking=True)
            batch_y = batch_y.cuda(local_rank, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(batch_x)
                loss = loss_fn(output, batch_y)
            acc1, acc5 = accuracy(output, batch_y, topk=(1, 5))

            dist.barrier()
            reduced_loss = reduce_mean(loss, world_size)
            reduced_acc1 = reduce_mean(acc1, world_size)
            reduced_acc5 = reduce_mean(acc5, world_size)

            losses.update(reduced_loss.item(), batch_y.size(0))
            top1.update(reduced_acc1.item(), batch_y.size(0))
            top5.update(reduced_acc5.item(), batch_y.size(0))
    if local_rank == 0:
        print(f"loss: {losses.val}, top1: {top1.val}, top5: {top5.val}")

    # ============ save checkpoint =============
    # you may want to save optimizer for resume
    if local_rank == 0:
        torch.save({'state_dict': model.state_dict()}, f"epoch_{epoch}.pth")
