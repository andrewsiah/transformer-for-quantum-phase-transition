import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


# Assuming the npy files are named 'delta_0.npy', 'delta_1.npy', etc.

# start_n = 0
# end_n = 4000
device = "cuda:0"
number_of_bins = 100000
num_epochs = 1000
eval_steps = 10
sequence_length = 10
batch_size = 200


import pickle

dir_path = "/shared/share_mala/andrew/diss/szhalf_L12_delta_-2_to_2_interval_4000_secondDelta_1_to_1_interval_1_data"
pickle_file_path = f"{dir_path}/{number_of_bins}_binned_values.pkl"

# Define the path to save the pickle file
# pickle_file_path = "/shared/share_mala/andrew/diss/szhalf_L12_delta_-2_to_2_interval_40000_secondDelta_1_to_1_interval_1_data_old/7500_binned_values.pkl"
# pickle_file_path = '/user/as6154/dissert/L12_half_data/7500_binned_values.pkl'
with open(pickle_file_path, "rb") as f:
    binned_values_loaded = pickle.load(f)

print(f"Binned values loaded from pickle file (first 10): {binned_values_loaded[:10]}")


class HighDimensionalDataset(Dataset):
    def __init__(self, data, n_items=1000, random=False):
        self.data = data
        self.random = random

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.random:
            start_index = np.random.randint(0, len(self.data) - sequence_length)
        else:
            start_index = min(idx, len(self.data) - sequence_length)
        sequence = self.data[start_index : start_index + sequence_length]
        return torch.tensor(sequence, dtype=torch.float32)


train_data = binned_values_loaded
test_data = binned_values_loaded

train_dataset = HighDimensionalDataset(train_data, n_items=1000, random=True)
test_dataset = HighDimensionalDataset(test_data, n_items=1000, random=False)


# Create dataloaders for training and testing
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

from transformers import GPT2Config, GPT2Model
import torch.nn as nn

print("Building model...")


class Transformer(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=4096, n_layer=12, n_head=4):
        super(Transformer, self).__init__()
        configuration = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self.n_dims = n_dims
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, n_dims)

    def forward(self, x):
        embeds = self._read_in(x)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        return prediction


n_dims = train_dataset[0].shape[1]
n_positions = 1024

model = Transformer(n_dims, n_positions)

input_data = torch.randn(batch_size, sequence_length, n_dims)

import wandb
from transformers import AdamW, get_linear_schedule_with_warmup
import pandas as pd

print("Setting up wandb...")

wandb.init(project="GPT_QFT", name=f"seq_len={sequence_length},batch_size={batch_size}")
loss_fn = nn.MSELoss()


optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=len(train_dataloader) * num_epochs,
)

from datetime import datetime

current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
loss_fp = f"loss_data/loss_data_{current_datetime}.csv"
config_fp = f"loss_data/config_{current_datetime}.txt"

with open(config_fp, "w") as config_file:
    config_file.write(f"number_of_bins={number_of_bins}\n")
    config_file.write(f"n_dims={n_dims}\n")
    config_file.write(f"batch_size={batch_size}\n")
    config_file.write(f"sequence_length={sequence_length}\n")
    config_file.write(f"learning_rate={1e-4}\n")
    config_file.write(f"num_warmup_steps={100}\n")
    config_file.write(f"num_training_steps={len(train_dataloader) * num_epochs}\n")

def evaluate_model(model, test_dataloader, device, loss_fn, loss_fp):
    loss_df = pd.DataFrame()
    model.eval()
    with torch.no_grad():
        loss_array = []
        for i, (test_input_seq) in enumerate(test_dataloader):
            test_target_seq = test_input_seq[:, -1:, :]
            test_input_seq = test_input_seq.to(device)
            test_target_seq = test_target_seq.to(device)

            test_output = model(test_input_seq)
            test_output = test_output[:, -1:, :]

            test_loss = loss_fn(test_output, test_target_seq)
            loss_array.append((i, test_loss.item()))
        loss_df = pd.DataFrame(loss_array, columns=["Index", "Loss"])
        loss_df.drop(columns=["Index"], inplace=True)
        loss_df = loss_df.transpose()
        import os

        if not os.path.exists(loss_fp):
            with open(loss_fp, "w") as f:
                pass
        loss_df.to_csv(loss_fp, mode="a", header=False, index=False)

        loss_array.sort(key=lambda x: x[1], reverse=True)
        print(f"Highest Loss at index: {loss_array[0][0]}, Loss: {loss_array[0][1]}")
        print(f"Lowest Loss at index: {loss_array[-1][0]}, Loss: {loss_array[-1][1]}")
        wandb.log(
            {
                "Highest Loss Index": loss_array[0][0],
                "Lowest Loss Index": loss_array[-1][0],
            }
        )
    model.train()


print("Training...")

steps_iterated = 0
for epoch in range(num_epochs):
    print(f"Epoch {epoch}")
    for i, (input_seq) in enumerate(tqdm(train_dataloader, desc="Training Iterations")):

        target_seq = input_seq[:, 1:, :]

        model.to(device)
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)
        output = model(input_seq)
        output = output[:, 1:, :]

        loss = loss_fn(output, target_seq)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        steps_iterated += 1

        if steps_iterated % eval_steps == 0:
            evaluate_model(model, test_dataloader, device, loss_fn, loss_fp)
            print(
                f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_dataloader)}], Loss: {loss.item()}"
            )
            wandb.log({"Loss": loss.item()})
