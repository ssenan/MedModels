from re import I
import torch
import torch.nn as nn
from torchvision import transforms

from DatasetLoader import sample_data
from utils import WGANP

class PGGAN:
    def __init__(self, generator, discriminator, device):
        super().__init__()
        self.gen = generator.to(device)
        self.dis = discriminator.to(device)
        self.depth = generator.depth
        self.latent_size = generator.latent_size
        self.device = device

    
    def sample_data(self, dataloader, depth):
        transform = transforms.Resize(2 ** (depth))

        loader = dataloader(transform)

        return loader

    def train(self, epochs, gen_lr, dis_lr, fade_in, start_depth, loader, 
            checkpoint_factor, save_dir):
        # Setting up dataset that that it can be iterated through

        model_dir, output_dir = save_dir/'models', save_dir/'outputs'
        model_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Defining optimizer for generator and discriminator
        gen_optim = torch.optim.Adam(
            params= self.gen.parameters(),
            lr = gen_lr, 
            betas=[0.0, 0.99], 
            eps=1e-8)
        
        dis_optim = torch.optim.Adam(
            params = self.dis.parameters(),
            lr = dis_lr, 
            betas=[0.0, 0.99], 
            eps=1e-8)

        # Instantiating current iteration/step and alpha variable
        alpha = 0
        iteration = 0
        for current_depth in range(start_depth, self.depth + 1):
            loader = sample_data(loader, 2 ** current_depth)
            dataset = iter(loader)

            current_res = int(2**current_depth)
            print(f"\nCurrent image depth: {current_depth}")
            print(f"Current Resolution: {current_res} x {current_res} x {current_res}")
            
            depth_index = current_depth - 1
            # Loading data at current depth
            real_image, label = next(dataset)
            #real_image = next(dataset)
            iteration += 1

            for epoch in range(1, epochs[depth_index] + 1):
                print(f"\nCurrent epoch: {epoch}")
                # Creating fade point to merge between resolutions
                total_batches = len(dataset)/2 
                fade_point = int(
                    fade_in[depth_index]
                    * epochs[depth_index]
                    * total_batches
                )

                # Calculate alpha to fade in layers
                alpha = iteration / fade_point if iteration < fade_point else 1

                # Training discriminator
                print(len(real_image))
                im_size = real_image.size(0)
                real_image = real_image.to(self.device)
                label = label.to(self.device)
                
                # Generate noise and fake image
                noise = torch.randn(im_size, self.latent_size).to(self.device)
                fake_image = self.gen(noise, current_depth, alpha).detach()
                
                # Calculating loss
                print("\nCalculating Discirminator loss")
                dis_loss = WGANP()
                #dis_loss = WGANP.dis_loss(
                    #discriminator=self.dis, real_image=real_image, fake_image=fake_image, depth=current_depth, alpha=alpha, labels=label)
                dis_loss.dis_loss(
                    self.dis, real_image, fake_image, current_depth, alpha
                    )
                dis_optim.zero_grad()
                dis_loss.backward()
                dis_optim.step()

                # Training generator
                print("\nCalculating Generator Loss")
                gen_loss = WGANP.gen_loss(fake_image, current_depth, alpha, label)
                gen_optim.zero_grad()
                gen_loss.backward()
                gen_optim.step()

                """
                WRITE SAVING MODEL AND IMAGE FUNCTION 
                
                """
                if (
                    epoch % checkpoint_factor == 0
                    or epoch == epochs[depth_index]
                ):
                    gen_save_file = model_dir / f"generator_depth_{current_depth}_epoch{epoch}.bin"
                    dis_save_file = model_dir / f"discrimator_depth_{current_depth}_epoch{epoch}.bin" 
                    torch.save(self.gen.state_dict(), gen_save_file)
                    torch.save(self.dis.state_dict(), dis_save_file)
        
        print("Training complete")
