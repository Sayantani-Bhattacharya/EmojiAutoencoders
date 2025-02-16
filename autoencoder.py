import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision.transforms as transforms
from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt

# ---------------------------
#  Hyperparameters
# ---------------------------

# 1. number of layers in encoder and decoder
# 2. size of each layer.
# 3. stride, kernel size of each of them.
# 4. regularization: dropout. 
# 5. probability of dropout.
# 6. latenet dimension.
# 7. activation function.: reLu.
# 8. learning rate.: 1e-3
# 9. weight decay: 1e-5
# 10. num_epochs = 20

# ---------------------------
# 1. Data Loading and Filtering
# ---------------------------

# Load the Hugging Face Emoji dataset.
hf_dataset = load_dataset("valhalla/emoji-dataset")

# Keyword to filter the images.
keyword = "face"
def filter_fn(example):
    return keyword.lower() in example["text"].lower()
filtered_dataset = hf_dataset["train"].filter(filter_fn)

# ---------------------------
# 2. Create a PyTorch Dataset Wrapper
# ---------------------------

class EmojiDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        if self.transform:
            image = self.transform(image)
        return image

# Define image transforms:
# - Resize images to 64x64.
# - Convert them to tensors and normalize to [0,1]
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Create final dataset.
emoji_dataset = EmojiDataset(filtered_dataset, transform=transform)

# ---------------------------
# 3. Split the Dataset (60/20/20)
# ---------------------------

dataset_length = len(emoji_dataset)
n_train = int(0.6 * dataset_length)
n_val = int(0.2 * dataset_length)
n_test = dataset_length - n_train - n_val

train_dataset, val_dataset, test_dataset = random_split(emoji_dataset, [n_train, n_val, n_test])

# Create DataLoaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ---------------------------
# 4. Convolutional Autoencoder
# ---------------------------

class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=128):  
        super(ConvAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder: Input size (3, 64, 64)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),   # -> (32, 32, 32)
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (64, 16, 16)
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # -> (128, 8, 8)
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),# -> (256, 4, 4)
            nn.ReLU(True)
        )
        
        # Flatten and compress to latent vector
        self.fc1 = nn.Linear(256 * 4 * 4, latent_dim)
        # Regularization: Dropout layer
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(latent_dim, 256 * 4 * 4)
        
        # Decoder: Mirrors the encoder using ConvTranspose2d
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (128, 8, 8)
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),   # -> (64, 16, 16)
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),    # -> (32, 32, 32)
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),     # -> (3, 64, 64)
            nn.Sigmoid()  # Ensure outputs are between 0 and 1
        )
    
    def forward(self, x):
        # Encode
        x_enc = self.encoder(x)
        x_enc = x_enc.view(x_enc.size(0), -1)
        latent = self.fc1(x_enc)
        latent = self.dropout(latent)  # Applying dropout for regularization
        
        # Decode
        x_dec = self.fc2(latent)
        x_dec = x_dec.view(x_dec.size(0), 256, 4, 4)
        x_recon = self.decoder(x_dec)
        return x_recon

# ---------------------------
# 5. Training Setup
# ---------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvAutoencoder(latent_dim=128).to(device)  # Adjust latent_dim as needed

# MSE loss
criterion = nn.MSELoss()

# Adam optimizer with weight decay (L2 regularization)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# ---------------------------
# 6. Training and Validation Loops
# ---------------------------

num_epochs = 20  
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
    epoch_train_loss = train_loss / len(train_loader)
    train_losses.append(epoch_train_loss)
    
    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            output = model(data)
            loss = criterion(output, data)
            val_loss += loss.item() * data.size(0)
    epoch_val_loss = val_loss / len(val_loader)
    val_losses.append(epoch_val_loss)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

# ---------------------------
# 7. Plot Learning Curves
# ---------------------------

plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Learning Curves')
plt.legend()
plt.show()

# ---------------------------
# 8. Evaluate on Test Set
# ---------------------------

model.eval()
total_test_loss = 0.0
with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        output = model(data)
        loss = criterion(output, data)
        total_test_loss += loss.item() * data.size(0)
final_test_loss = total_test_loss / len(test_dataset)
print(f"Final Average Test MSE: {final_test_loss:.4f}")

# ---------------------------
# 9. Side-by-Side Example of 5 Input and Output Images
# ---------------------------

# Get one batch from the test loader
data_iter = iter(test_loader)
images = next(data_iter)
images = images.to(device)
with torch.no_grad():
    reconstructions = model(images)

# Move to CPU and convert to numpy arrays for plotting
images_np = images.cpu().numpy().transpose(0, 2, 3, 1)
recon_np = reconstructions.cpu().numpy().transpose(0, 2, 3, 1)

# Plot the first 5 images and their reconstructions
num_examples = 5
fig, axes = plt.subplots(2, num_examples, figsize=(15, 6))
for i in range(num_examples):
    # Original image
    axes[0, i].imshow(images_np[i])
    axes[0, i].axis('off')
    # Reconstructed image
    axes[1, i].imshow(recon_np[i])
    axes[1, i].axis('off')

axes[0, 0].set_title('Original')
axes[1, 0].set_title('Reconstructed')
plt.suptitle("Side-by-Side Input (Top) and Output (Bottom) Examples")
plt.show()
