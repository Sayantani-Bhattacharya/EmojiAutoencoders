import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision.transforms as transforms
from datasets import load_dataset, concatenate_datasets
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
# 8. learning_rate = 0.001
# 9. batch_size = 64
# 10. hidden_dim = 128
# 11. num_epochs = 20
# 12. lambda_classification = 0.5  


# ---------------------------
# 0. Data Loading and Filtering
# ---------------------------

# Load the Hugging Face Emoji dataset.
hf_dataset = load_dataset("valhalla/emoji-dataset")
keyword = "face"
def filter_fn(example):
    return keyword.lower() in example["text"].lower()
filtered_dataset = hf_dataset["train"].filter(filter_fn)

# ---------------------------
# 1. Classifying the data
# ---------------------------

class_mapping = {
    "happy face": 0, "grinning face": 0, "smiling face": 0,   # Class 0: Happy
    "sad face": 1, "crying face": 1, "frowning face" : 1       # Class 1: Sad
}

def assign_class(example):
    description = example["text"].lower()
    for key, label in class_mapping.items():
        if key in description:
            return label
    return -1  # Unknown or unlabeled cases

filtered_dataset = filtered_dataset.map(lambda x: {"class_label": assign_class(x)})
# Remove unlabeled data
filtered_dataset = filtered_dataset.filter(lambda x: x["class_label"] != -1)  

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
        # class_label = item["class_label"]
        class_label = torch.tensor(self.dataset[idx]["class_label"], dtype=torch.long)
       
        if self.transform:
            image = self.transform(image)
        return image, class_label

    @classmethod
    def concatenate(cls, dataset1, dataset2, transform=None):
        """
        Concatenates two EmojiDataset instances into one.
        
        Args:
            dataset1 (EmojiDataset): The first dataset instance.
            dataset2 (EmojiDataset): The second dataset instance.
            transform (callable, optional): The transform to apply. If not provided,
                dataset1's transform will be used.
        
        Returns:
            EmojiDataset: A new EmojiDataset instance with the concatenated data.
        """
        # Concatenate the underlying Hugging Face datasets
        new_hf_dataset = concatenate_datasets([dataset1.dataset, dataset2.dataset])
        
        # Decide which transform to use
        new_transform = transform if transform is not None else dataset1.transform
        
        # Return a new instance of EmojiDataset with the concatenated dataset
        return cls(new_hf_dataset, transform=new_transform)
    

# Define image transforms:
# - Resize images to 64x64.
# - Convert them to tensors and normalize to [0,1]
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])
# Create final dataset.
original_dataset = EmojiDataset(filtered_dataset, transform=transform)


# ---------------------------
# 3. Add Data Augmentation
# ---------------------------

# Define data augmentation transformations
data_augmentation = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
])
augmented_datasets = []
previous_dataset = original_dataset #Just a placeholder value.

# Run augmentation 4 times to create 4 copies per original image.
num_copies = 50
for i in range(num_copies):
    aug_dataset = EmojiDataset(filtered_dataset, transform=data_augmentation)
    if (i > 0):
        augmented_datasets = EmojiDataset.concatenate(previous_dataset,aug_dataset)
        previous_dataset = augmented_datasets
    else:
        previous_dataset = aug_dataset  
    
final_dataset = EmojiDataset.concatenate(original_dataset,augmented_datasets)
print("/////////////////////////////////")
print(f"Original dataset size: {len(original_dataset)}")
print(f"Augmented dataset size: {len(augmented_datasets)}")
print(f"Final dataset size: {len(final_dataset)}")


# ---------------------------
# 3. Split the Dataset (60/20/20)
# ---------------------------


dataset_length = len(final_dataset)
n_train = int(0.6 * dataset_length)
n_val = int(0.2 * dataset_length)
n_test = dataset_length - n_train - n_val

train_dataset, val_dataset, test_dataset = random_split(final_dataset, [n_train, n_val, n_test])

# Create DataLoaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# ---------------------------
# 4. Multitask Autoencoder
# ---------------------------

import torch.nn as nn

'''
Loss Functions:
    1. MSELoss() for regression.
    2. CrossEntropyLoss() for classification.
'''
class MultiTaskAutoencoder(nn.Module):
    def __init__(self, latent_dim=128, num_classes=2):
        super(MultiTaskAutoencoder, self).__init__()
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
        
        # Classification
        self.classifier = nn.Linear(latent_dim, num_classes)

    def forward(self, x):
        latent = self.encoder(x)
        latent = latent.view(latent.shape[0], -1)
        latent = self.fc1(latent)

        reconstructed = self.fc2(latent)  # Map back to (256 * 4 * 4)
        reconstructed = reconstructed.view(latent.shape[0], 256, 4, 4)
        reconstructed = self.decoder(reconstructed)

        classification = self.classifier(latent)
        return reconstructed, classification


# ---------------------------
# 5. Training Setup
# ---------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiTaskAutoencoder(latent_dim=128).to(device)  

# MSE loss
criterion_mse = nn.MSELoss()
criterion_classification = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10 #20
learning_rate = 0.001
lambda_classification = 0.5

# ---------------------------
# 6. Training and Validation Loops
# ---------------------------

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for batch in train_loader:
        features, class_labels = batch  # Extract features & labels
        features = features.to(device)
        class_labels = class_labels.to(device)       
        optimizer.zero_grad()
        reconstructed, pred_class = model(features)
        loss_mse = criterion_mse(reconstructed, features)  # Autoencoder loss
        loss_classification = criterion_classification(pred_class, class_labels)  # Classification loss
        # Combine losses
        loss = loss_mse + lambda_classification * loss_classification
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * features.size(0)
        # Calculate training accuracy
        _, predicted = torch.max(pred_class, 1)
        correct = (predicted == class_labels).sum().item()
        epoch_train_accuracy = correct / len(class_labels)
    train_accuracies.append(epoch_train_accuracy)
    epoch_train_loss = train_loss / len(train_loader)
    train_losses.append(epoch_train_loss)

    # Validation step
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:            
            features, class_labels = batch  # Extract features & labels 
            features = features.to(device)
            class_labels = class_labels.to(device)      
            optimizer.zero_grad()
            reconstructed, pred_class = model(features)
            loss_mse = criterion_mse(reconstructed, features)  # Autoencoder loss
            loss_classification = criterion_classification(pred_class, class_labels)  # Classification loss
            # Combine losses
            loss = loss_mse + lambda_classification * loss_classification
            val_loss += loss.item() * features.size(0)
            # Calculate validation accuracy
            _, predicted = torch.max(pred_class, 1)
            correct = (predicted == class_labels).sum().item()
            epoch_val_accuracy = correct / len(class_labels)
    val_accuracies.append(epoch_val_accuracy)
    epoch_val_loss = val_loss / len(val_loader)
    val_losses.append(epoch_val_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Train Accuracy: {epoch_train_accuracy:.4f}, Val Accuracy: {epoch_val_accuracy:.4f}")


# ---------------------------
# 7. Plot Learning Curves
# ---------------------------


epochs = range(num_epochs)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Learning Curves')
plt.legend()
# plt.savefig("results2/MSEplot.png")

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs+1), train_accuracies, label="Classification Training Accuracy")
plt.plot(range(1, num_epochs+1), val_accuracies, label="Classification Validation Accuracy")
plt.xlabel('Epoch')
plt.ylabel("Accuracy")
plt.title('Learning Curves')
plt.legend()
plt.savefig("results2/plot.png")


# ---------------------------
# 8. Evaluate on Test Set
# ---------------------------

model.eval()
total_test_loss = 0.0
test_loss = 0.0
with torch.no_grad():
    for batch in test_loader:
        features, class_labels = batch  # Extract features & labels   
        features = features.to(device)
        class_labels = class_labels.to(device)    
        optimizer.zero_grad()
        reconstructed, pred_class = model(features)
        loss_mse = criterion_mse(reconstructed, features)  # Autoencoder loss
        loss_classification = criterion_classification(pred_class, class_labels)  # Classification loss
        # Combine losses
        loss = loss_mse + lambda_classification * loss_classification
        test_loss += loss.item() * features.size(0)
        # Calculate test accuracy
        _, predicted = torch.max(pred_class, 1)
        correct = (predicted == class_labels).sum().item()
        test_accuracy = correct / len(class_labels)

final_test_loss = test_loss / len(test_loader)
final_test_accuracy = test_accuracy
print(f"Final Average Test MSE: {final_test_loss:.4f}, Test Accuracy: {final_test_accuracy:.4f}")



# ---------------------------
# 9. Side-by-Side Example of 5 Input and Output Images
# ---------------------------

with torch.no_grad():
    for batch in test_loader:
            features, class_labels = batch  # Extract features & labels   
            features = features.to(device)
            class_labels = class_labels.to(device)    
            optimizer.zero_grad()
            reconstructed, pred_class = model(features)

# Move to CPU and convert to numpy arrays for plotting
images_np = features.cpu().numpy().transpose(0, 2, 3, 1)
recon_np = reconstructed.cpu().numpy().transpose(0, 2, 3, 1)

original_labels = class_labels.cpu().numpy()
predicted_labels = pred_class.argmax(dim=1).cpu().numpy()
label_mapping = {0: "Happy Face", 1: "Sad Face"}
# Plot the first 5 images and their reconstructions
num_examples = 5
fig, axes = plt.subplots(2, num_examples, figsize=(15, 6))
for i in range(num_examples):
    # Original image
    axes[0, i].imshow(images_np[i])
    axes[0, i].axis('off')
    axes[0, i].set_title(f"Original: {label_mapping[original_labels[i]]}")
    # Reconstructed image
    axes[1, i].imshow(recon_np[i])
    axes[1, i].axis('off')
    axes[1, i].set_title(f"Pred: {label_mapping[predicted_labels[i]]}")

axes[0, 0].set_title('Original')
axes[1, 0].set_title('Reconstructed')
plt.suptitle("Side-by-Side Input (Top) and Output (Bottom) Examples")
plt.savefig("results2/autoencoder_results.png")









