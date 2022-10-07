#from ddpm_unet import Unet
from unet import Unet
from diffusion import Diffusion, beta_schedule
from DatasetLoader import get_data_noarg

import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def noise_estimation_loss(model,
                          x0:torch.Tensor, 
                          t:torch.LongTensor,
                          e:torch.Tensor,
                          b:torch.Tensor,
                          keepdim=False):
    a = torch.cumprod(1-b,dim=0).index_select(0, t).view(-1,1, 1, 1, 1).to(device)
    x = x0 * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(x, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3,4))
    else:
        return (e - output).square().sum(dim=(1, 2, 3,4)).mean(dim=0)

def model_predict(x0, beta, t, *, status='train', transpose=True):
    with torch.no_grad():
        a = torch.cumprod(1-beta, dim=0)
        a_t = a[t].to(device)
        e = torch.rand_like(x0).to(device)

        if status == "train":
            xt = torch.sqrt(1-a_t)*e + torch.sqrt(a_t)*x0

        elif status == "test":
            xt = x0

        else:
            print("No status defined")

        e_hat = model(xt, torch.from_numpy(np.array([T])).to(device))
        x0_hat = 1 / torch.sqrt(a_t) * xt - torch.sqrt(1-a_t) / torch.sqrt(a_t) * e_hat
        
        if transpose:
            x0_show = np.transpose(x0[0,0,:,:].detach().cpu().numpy())
            x0_hat_show = np.transpose(x0_hat[0,0,:,:].detach().cpu().numpy())
            xt_show = np.transpose(xt[0,0,:,:].detach().cpu().numpy())

        else:
            x0_show = x0[0,0,:,:].detach().cpu().numpy()
            x0_hat_show = x0_hat[0,0,:,:].detach().cpu().numpy()
            xt_show = xt[0,0,:,:].detach().cpu().numpy()

        return x0_show, x0_hat_show, xt_show

global T
T = 100
n_epochs = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_dir = "./ddpm/dataset/train_output"

train_data = get_data_noarg("./dataset", batch_size=1)
beta_schedules = beta_schedule("cosine", beta_start=0.0001, beta_end=0.003, timestep=T)
#plt.plot(beta_schedules)
beta_schedules = beta_schedules.float().to(device)

model = Unet().to(device)

lr = 1e-4
optim = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
#


for epoch in range(n_epochs):
    values  = range(len(train_data))

    with tqdm(len(values)) as pbar:
        for step, x in enumerate(train_data):

            model.train()

            x = x.to(device)
            e = torch.rand_like(x).to(device)
            b = beta_schedules

            n = x.size(0)
            t = torch.randint(low=0, high=T, size=(n//2 + 1,))
            t = torch.cat([t, T-t-1], dim=0)[:n].to(device)

            loss = noise_estimation_loss(model, x, t, e, b)
            print(loss.shape)
            #prediction = model(x,t)
            #loss = torch.nn.CrossEntropyLoss(prediction)
            optim.zero_grad()
            loss.backward()
            optim.step()

            pbar.update(1)
            pbar.set_description(f"Epoch: {epoch+1}. Loss value: {loss}")

            if step % (len(train_data)-1) == 0 and step != 0:
                timestep = random.randint(1,T-1)
                x0,x_pred,xt = model_predict(x,b,timestep,status='train',transpose=False)
                
                plt.figure(figsize=(15,6))
                plt.subplot(1,3,1),plt.imshow(x0[:,:500],cmap='gray'),plt.axis('off'),plt.title('noisy')
                plt.subplot(1,3,2),plt.imshow(x_pred[:,:500],cmap='gray'),plt.axis('off'),plt.title('sample')
                plt.subplot(1,3,3),plt.imshow(xt[:,:500],cmap='gray'),plt.axis('off'),plt.title('t={}'.format(timestep))
                plt.show()

    if epoch % 20 == 0 and epoch != 0:
            name = 'DDPM_oct_dataset2_gt=sf.pt'
            torch.save(model.state_dict(),model_dir+name)
name = 'DDPM_oct_dataset2_gt=sf.pt'
torch.save(model.state_dict(),model_dir+name)    
