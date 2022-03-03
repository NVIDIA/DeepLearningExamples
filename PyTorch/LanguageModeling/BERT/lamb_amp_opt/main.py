import copy

import apex
import torch

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification, make_regression

from fused_lamb import FusedLAMBAMP

N_SAMPLES = 10000
N_FEATURES = 32
BATCH_SIZE = 100


def print_param_diff(optimizer):
    with torch.no_grad():
        for i, (group, master_group) in enumerate(zip(optimizer.param_groups, optimizer.param_groups_fp32)):
            for ii, (p, master_p) in enumerate(zip(group['params'], master_group['params'])):
                diff = (p - master_p.half()).float().abs().mean().item()
                print(f"  {i}th group, {ii}th param diff: {diff}")


class TestMod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(N_FEATURES, N_FEATURES // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(N_FEATURES // 2, 2),
        )
    def forward(self, inputs) :
        return self.layers(inputs)


def main() :
    loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
    model = TestMod()
    model.cuda()
    model.half()
    model.train()
    model = torch.jit.script(model)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta', 'LayerNorm']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
    optimizer = FusedLAMBAMP(optimizer_grouped_parameters)

    x, y = make_classification(n_samples=N_SAMPLES, n_features=N_FEATURES, random_state=0)
    x = StandardScaler().fit_transform(x)
    inputs = torch.from_numpy(x).cuda().half()
    targets = torch.from_numpy(y).cuda().long()
    del x, y
    dataset = torch.utils.data.TensorDataset(inputs, targets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)
    for epoch in range(20):
        loss_values = []
        for i, (x, y) in enumerate(loader):
            with torch.cuda.amp.autocast():
                out1 = model(x)
                # Might be better to run `CrossEntropyLoss` in
                # `with torch.cuda.amp.autocast(enabled=False)` context.
                out2 = loss(out1, y)
            grad_scaler.scale(out2).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            optimizer.zero_grad(set_to_none=True)
            loss_values.append(out2.item())

        print(f"Epoch: {epoch}, Loss avg: {np.mean(loss_values)}")
        print_param_diff(optimizer)

    print("state dict check")
    optimizer.load_state_dict(optimizer.state_dict())
    print_param_diff(optimizer)

if __name__ == '__main__':
    main()
