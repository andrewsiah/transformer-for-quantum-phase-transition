import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

class WavefunctionDataset(Dataset):
    def __init__(self, file_prefix, start_n, end_n):
        self.file_prefix = file_prefix
        self.start_n = start_n
        self.end_n = end_n
        
    def __len__(self):
        return self.end_n - self.start_n
    
    def __getitem__(self, idx):
        file_path = f'{self.file_prefix}{self.start_n + idx}.npy'
        wavefunction = np.load(file_path)
        return torch.from_numpy(wavefunction).float()

# Set the file prefix and range
file_prefix = '/Users/asiah/Documents/columbia/dissertation/half_data/delta_'
start_n = 0
end_n = 799

# Create the dataset
dataset = WavefunctionDataset(file_prefix, start_n, end_n)

# Create the DataLoader
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Load the MPT model and tokenizer
model_name = "mosaicml/mpt-7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Fine-tune the MPT model on the wavefunction data
model.train()
for batch in dataloader:
    # Tokenize the wavefunction data
    inputs = tokenizer([str(item) for item in batch], return_tensors="pt", padding=True)
    # inputs = tokenizer(batch.tolist(), return_tensors="pt", padding=True)
    
    # Forward pass
    outputs = model(**inputs, labels=inputs["input_ids"])
    
    # Backward pass and optimization step
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Save the fine-tuned model
model.save_pretrained("fine_tuned_mpt")

