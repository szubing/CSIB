import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

dim = 10000
h_dim = 0
n_sample = dim
n_sample = int(n_sample)
clear = True
total_epoch = 5000

def get_dataset(n_sample, dim):
    X = 2*torch.rand(n_sample, dim) - 1
    Y = (torch.rand(n_sample)>0.5).float()
    return X, Y.unsqueeze(-1)

def get_model(dim, h_dim, lr=0.1, wd=0):
    if h_dim == 0:
        network = torch.nn.Linear(dim, 1)
    elif type(h_dim) == list:
        modules = []
        for dim_ in h_dim:
            modules.append(torch.nn.Linear(dim, dim_))
            dim = dim_
        modules.append(torch.nn.Linear(dim,1))
        network = torch.nn.Sequential(*modules)
    else:
        raise ValueError('h_dim is not valid')
    optimizer = torch.optim.SGD(
                    network.parameters(),
                    lr=lr,
                    weight_decay=wd)
    sche = torch.optim.lr_scheduler.StepLR(optimizer, step_size=total_epoch//2, gamma=0.1)
    return network, optimizer, sche

def train(epoch=total_epoch):
    network, optimizer, sche = get_model(dim, h_dim)
    network = network.cuda()
    X, Y = get_dataset(n_sample, dim)
    X, Y = X.cuda(), Y.cuda()
    criteria = torch.nn.BCEWithLogitsLoss()
    losses = []
    errors = []
    for e in tqdm(range(epoch)):
        optimizer.zero_grad()
        out = network(X)
        loss = criteria(out, Y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        error = out.gt(0).ne(Y).float().mean().item()
        errors.append(error)
        print(loss.item(), '  --  ', error)
        sche.step()
    
    return losses, errors


if not clear and os.path.exists('loss{}_n{}.pt'.format(dim,n_sample)):
    losses = torch.load('loss{}_n{}.pt'.format(dim,n_sample))
    errors = torch.load('error{}_n{}.pt'.format(dim,n_sample))
else:
    os.makedirs('imgs', exist_ok=True)
    errs=[]
    for seed in range(100):
        losses, errors = train()
        errs.append(errors[-2])
        torch.save(losses, 'imgs/loss{}_n{}_seed{}.pt'.format(dim,n_sample,seed))
        torch.save(errors, 'imgs/error{}_n{}_seed{}.pt'.format(dim,n_sample,seed))
      
        fig, ax = plt.subplots(figsize=(5, 4), layout='constrained')
        #ax.plot(losses, label='loss')  # Plot some data on the axes.
        ax.plot(errors, label='error')  # Plot more data on the axes...
        ax.set_xlabel('epoch')  # Add an x-label to the axes.
        ax.set_ylabel('error')  # Add a y-label to the axes.
        ax.set_title("training error")  # Add a title to the axes.
        #ax.legend();  # Add a legend.
        plt.axhline(y=0.0, color='r', linestyle='--')
        plt.savefig("imgs/error{}_n{}_seed{}.pdf".format(dim,n_sample,seed), format="pdf", bbox_inches="tight")
    torch.save(errs, 'errs.pt')
    print(errs)