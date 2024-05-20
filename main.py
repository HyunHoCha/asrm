import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import MLP
from MLP import *
import utils
from utils import *
import pickle

# python3 main.py

mlp_ratio = 3
batch_size = 50
learning_rate = 0.001
epochs = 20
C = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = int(C * (C + 3) / 2)  # {C * (C - 1) / 2} + C (probs) + C (purities)
hidden_size1 = input_size * mlp_ratio
hidden_size2 = hidden_size1
output_size = C

model = MLP_net_noNorm_double(input_size, hidden_size1, hidden_size2, output_size, C)

model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# load training data
with open('train/dim2cls' + str(C) + 'InnerDistancePurity.pkl', 'rb') as f:
    loaded_data = pickle.load(f)
fid_prob_sig_success = [(torch.cat((torch.tensor(datum[2]).float(), torch.from_numpy(datum[1]).float(), torch.tensor(datum[4]).float()), dim=0), datum[0], datum[3]) for datum in loaded_data]
dataset = FidProbData(fid_prob_sig_success)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# load validation data
with open('val/dim2cls' + str(C) + 'InnerDistancePurity.pkl', 'rb') as f:
    loaded_data_val = pickle.load(f)
fid_prob_sig_success_val = [(torch.cat((torch.tensor(datum[2]).float(), torch.from_numpy(datum[1]).float(), torch.tensor(datum[4]).float()), dim=0), datum[0], datum[3]) for datum in loaded_data_val]
dataset_val = FidProbData(fid_prob_sig_success_val)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)





# train
for epoch in range(epochs):
    for step, data in enumerate(dataloader):
        optimizer.zero_grad()

        input_data, sigma_list_batch, PGM_prob_batch = data
        input_data = input_data.to(device)
        sigma_list_batch = [tensor.to(device) for tensor in sigma_list_batch]
        PGM_prob_batch = PGM_prob_batch.to(device)
        weights = model(input_data)

        loss = torch.tensor(0, dtype=torch.complex128).to(device)
        mean_PGM = torch.tensor(0, dtype=torch.complex128).to(device)
        for idx in range(batch_size):
            sigma_list = [sigma_list_batch[i][idx] for i in range(C)]
            loss -= better(sigma_list, input_data[idx][int(C * (C - 1) / 2):int(C * (C + 1) / 2)], weights[idx])  # negative success probability
            loss += PGM_prob_batch[idx]  # positive PGM probability
            mean_PGM += PGM_prob_batch[idx]
        loss /= batch_size
        mean_PGM /= batch_size
        loss.backward()
        optimizer.step()

        if (step + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Step [{step + 1}/{len(dataloader)}], Loss: {loss.item():.4f}, average PGM: {mean_PGM.item():.4f}")
print("Training complete.")

save_path = "models/model.pth"
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")





# val
model = MLP_net_noNorm_double(input_size, hidden_size1, hidden_size2, output_size, C)
model = model.to(device)
saved_state_dict = torch.load("models/model.pth")
model.load_state_dict(saved_state_dict)
model.eval()
with torch.no_grad():
    total_loss = 0
    total_mean_PGM = 0
    total_mean_better = 0
    total_mean_optimal = 0

    for step, data in enumerate(dataloader_val):
        print('validation batch ', step)
        
        input_data, sigma_list_batch, PGM_prob_batch = data
        input_data = input_data.to(device)
        sigma_list_batch = [tensor.to(device) for tensor in sigma_list_batch]
        PGM_prob_batch = PGM_prob_batch.to(device)
        weights = model(input_data)

        loss = torch.tensor(0, dtype=torch.complex128).to(device)
        mean_PGM = torch.tensor(0, dtype=torch.complex128).to(device)
        mean_better = torch.tensor(0, dtype=torch.complex128).to(device)
        mean_optimal = torch.tensor(0, dtype=torch.complex128).to(device)
        for idx in range(batch_size):
            sigma_list = [sigma_list_batch[i][idx] for i in range(C)]
            probs = input_data[idx][int(C * (C - 1) / 2):int(C * (C + 1) / 2)]
            better_prob = better(sigma_list, probs, weights[idx])  # negative success probability
            loss -= better_prob
            loss += PGM_prob_batch[idx]  # positive PGM probability
            mean_PGM += PGM_prob_batch[idx]
            mean_better += better_prob
            mean_optimal += SDP([sigma.cpu() for sigma in sigma_list], probs.cpu())[0]

        loss /= batch_size
        mean_PGM /= batch_size
        mean_better /= batch_size
        mean_optimal /= batch_size

        total_loss += loss.item()
        total_mean_PGM += mean_PGM.item()
        total_mean_better += mean_better.item()
        total_mean_optimal += mean_optimal
    
    avg_loss = total_loss / len(dataloader_val)
    avg_mean_PGM = total_mean_PGM / len(dataloader_val)
    avg_mean_better = total_mean_better / len(dataloader_val)
    avg_mean_optimal = total_mean_optimal / len(dataloader_val)
    print(f"[Validation]  Average Loss: {avg_loss:.4f}, Average PGM: {avg_mean_PGM:.4f}, Average ASRM: {avg_mean_better:.4f}, Average optimal: {avg_mean_optimal:.4f}")
