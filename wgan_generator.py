import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler


# Environment Variables
datasets_folder = 'C:/ts_data/real_data/'  # Folder containing real datasets
synthetic_data_folder = 'C:/ts_data/synthetic_data/'  # Folder to store generated datasets
dataset_paths = glob.glob(os.path.join(datasets_folder, '*.csv'))

# Modifiable WGAN Parameters
column_of_interest = 'timeseries_column'  # Name of the column to use for training the model
sequence_length = 14000  # Length of the time series sequences to train the model and generate

latent_dim = 10  # Dimension of the latent space for the generator
batch_size = 1  # Batch size for training
epochs = 15  # Number of training epochs

n_critic = 8  # Number of discriminator updates per generator update
lambda_gp = 10.0  # Gradient penalty coefficient
gen_lr = 0.0005  # Learning rate for the generator
dis_lr = 0.0005  # Learning rate for the discriminator

betas=(0.9, 0.999) # For Adam optimizer


# Define the Generator and Discriminator networks
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),  # Output size is (batch_size, 14000)
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)


# Define the dataset loading function
def load_dataset(dataset_paths, sequence_length):
    dataset_list = []
    for dataset_path in dataset_paths:
        dataset = pd.read_csv(dataset_path, index_col=0, parse_dates=True)
        dataset = dataset.iloc[:sequence_length]
        dataset = dataset[[column_of_interest]].values
        scaler = MinMaxScaler()
        dataset = scaler.fit_transform(dataset)
        dataset_list.append(dataset)

    shape = [sequence_length]
    shape[:0] = [len(dataset_list)]
    
    # Stack the datasets vertically to get a matrix of size (nb of training datasets, 14000)
    dataset_matrix = np.concatenate(dataset_list).reshape(shape)

    # Transpose the dataset
    dataset_matrix = torch.from_numpy(dataset_matrix).float()
    
    return dataset_matrix, scaler

# Create a folder to save generated samples
os.makedirs(synthetic_data_folder, exist_ok=True)

# Define the generator, discriminator, and optimizers
generator = Generator(latent_dim, sequence_length)
discriminator = Discriminator(sequence_length)

optimizer_G = optim.Adam(generator.parameters(), lr=gen_lr, betas=betas)
optimizer_D = optim.Adam(discriminator.parameters(), lr=dis_lr, betas=betas)

# Training loop
for epoch in range(epochs):
    real_data, scaler = load_dataset(dataset_paths, sequence_length)
    dataloader = DataLoader(real_data, batch_size, shuffle=False)

    for i, data_batch in enumerate(dataloader):
        real_samples = data_batch

        # Train the discriminator
        for _ in range(n_critic):
            optimizer_D.zero_grad()

            # Sample noise for generator
            z = torch.randn(batch_size, latent_dim)

            # Generate fake samples
            fake_samples = generator(z).detach()

            # Calculate discriminator scores
            real_scores = discriminator(real_samples)
            fake_scores = discriminator(fake_samples)

            # Compute gradient penalty
            alpha = torch.rand(batch_size, 1)
            alpha = alpha.expand(real_samples.size())
            interpolates = alpha * real_samples + (1 - alpha) * fake_samples
            interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
            d_interpolates = discriminator(interpolates)
            gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                            grad_outputs=torch.ones(d_interpolates.size()),
                                            create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp

            # Compute WGAN loss
            d_loss = -torch.mean(real_scores) + torch.mean(fake_scores) + gradient_penalty

            d_loss.backward()
            optimizer_D.step()

        # Train the generator
        optimizer_G.zero_grad()
        z = torch.randn(batch_size, latent_dim)
        fake_samples = generator(z)
        fake_scores = discriminator(fake_samples)
        g_loss = -torch.mean(fake_scores)
        g_loss.backward()
        optimizer_G.step()

        if (i + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}], Batch [{i + 1}/{len(dataloader)}], "
                f"D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}"
            )

    # Save generated samples
    if (epoch + 1) % 1 == 0:
        with torch.no_grad():
            z = torch.randn(1, latent_dim)
            fake_samples = generator(z)
            generated_data = scaler.inverse_transform(fake_samples[0].reshape(-1,1).numpy())
            df = pd.DataFrame(generated_data, columns=[column_of_interest])
            save_path = os.path.join(synthetic_data_folder, f"generated_sample_epoch_{epoch + 1}.csv")
            df.to_csv(save_path, index=False, sep=';', decimal=',')

print("Training finished!")