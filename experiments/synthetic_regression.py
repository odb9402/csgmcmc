import glob
import copy, argparse, random, sys
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.style.use('bmh')
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import pdb
sys.path.append('..')
from utils.loss import gaussian_nll_loss
from utils.utils import mean_confidence_interval, confidence_interval
from models.simple import Model

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='1D synthetic data regression experiment')
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--interval', type=int, default=2000, help="Iteration interval to save checkpoints")
parser.add_argument('--temperature', type=float, default=1/400, help="Temperature, default: 1/datasize")
parser.add_argument('--curricular', action='store_true')
args = parser.parse_args()
device = 'cuda:2'

def create_features(X):
    X_square = X**2
    X_cos = np.cos(X)
    return np.stack((X, X_square, X_cos), axis=1).squeeze()

np.random.seed(7777)
func = lambda x: np.sin(x)*x - 5 

def features(x):
    return np.hstack([x[:, None] / 2.0, (x[:, None] / 2.0) ** 2])

data = np.load("data.npy")
x, y = data[:, 0], data[:, 1]
y = y[:, None]
f = features(x)
dataset = torch.utils.data.TensorDataset(torch.from_numpy(f.astype(np.float32)), 
                                         torch.from_numpy(y.astype(np.float32)))
loader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=True)

def noise_loss(lr,alpha=1.0):
    noise_loss = 0.0
    noise_std = (2/lr*alpha)**0.5
    for var in model.parameters():
        means = torch.zeros(var.size()).to(device)
        noise_loss += torch.sum(var * torch.normal(means, std = noise_std).to(device))
    return noise_loss

torch.manual_seed(0)
model = Model(input_dim=2).to(device)
opt = optim.SGD(model.parameters(), lr=1e-2, weight_decay=0.0)
scheduler = optim.lr_scheduler.LambdaLR(optimizer=opt,
                            lr_lambda=lambda epoch: 0.99 ** epoch,
                            last_epoch=-1,
                            verbose=False)
model.train()
for it in range(24000):
    X_train, Y_train = next(iter(loader))
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    m = model(X_train)
    loss = F.mse_loss(m, Y_train, reduction='none')
    
    if args.curricular and (it % 2000) + 1 < 1500:# and it % args.interval < args.interval//2:
        loss = torch.topk(loss, 15, dim=0, largest=False).values.mean()
    if args.curricular and (it % 2000) + 1 >= 1500:# and it % args.interval < args.interval//2:
        loss = torch.topk(loss, 15, dim=0).values.mean()
    else:
        loss = loss.mean()

    if it > 1000:# and it % args.interval < args.interval//2:
        loss_noise = noise_loss(scheduler.get_last_lr()[0])*(args.temperature/400)**0.5
        (loss + loss_noise).backward()
    else:
        loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    
    if it % 100 == 0:
        print(f'Step: {it} Loss: {loss.item():.3f}')
        scheduler.step()
    
    if it % args.interval == 0 and it != 0:
        torch.save(model.state_dict(), f'../sampled_models/synthetic_sgld/reg_model_{it}.ckpt')

##* Inference with ckpts 
preds = []
#stds = []
sampled_model_paths = glob.glob('../sampled_models/synthetic_sgld/*.ckpt')
z = np.linspace(-10, 10, 100)
inp = torch.from_numpy(features(z).astype(np.float32)).to(device)
for path in sampled_model_paths:
    model = Model(input_dim=2).to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    pred_y = model(inp.float())
    preds.append(pred_y)
    #stds.append(std_y.sqrt())

means = torch.stack(preds).detach().cpu().numpy()
#stds = torch.stack(stds).detach().cpu().numpy()
mean  = np.mean(means, axis = 0)
std = np.std(means, axis = 0) 

plt.figure()
plt.plot(z, mean)
for conf in [0.95, 0.70, 0.50, 0.30]:
    _, lower, upper = confidence_interval(mean, std, confidence=conf)
    plt.fill_between(z.flatten(), lower.flatten(), upper.flatten(), alpha=0.25)
plt.scatter(x, y)
plt.tight_layout()
plt.xlim(-10,10)
plt.ylim(-0.75,1.25)
if args.curricular:
    plt.savefig("pics/regpoly_sgld_curricular.png")
else:
    plt.savefig("pics/regpoly_sgld.png")