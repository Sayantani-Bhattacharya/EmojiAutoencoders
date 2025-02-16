import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision.models as models
import torchattacks
from torch.utils.data import DataLoader
import numpy as np
import io
from sklearn.preprocessing import LabelEncoder


class Dataset(Dataset):

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # print("//////////////////////////////////////////////")
        # print(type(face_emojis.iloc[0]["image"]))
        # print(face_emojis.iloc[0]["image"].keys())
        # print(face_emojis.iloc[0]["image"]["path"])
        
        image_data = self.dataframe.iloc[idx]['image']
        # label = self.dataframe.iloc[idx]['text']
        label = self.dataframe.iloc[idx]['encoded_label'] 

        image = Image.open(io.BytesIO(image_data["bytes"])).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long) 

# Define transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# input: DataFrame of the filtered dataset.
# input: Number of classes in the dataset.
# output: Augmented DataFrame with adversarial examples.
def augment_dataset(dataset):

    # Create dataset
    dataset = Dataset(dataset, transform=transform)


    # DataLoader for the dataset
    # Instead of processing one image at a time, it loads 32 images & labels together to speed up training.
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Lists to store adversarial images and labels
    adv_images = []
    adv_labels = []

    unique_labels = dataset.dataframe['text'].unique() 
    num_classes = len(unique_labels)


    # Load pretrained ResNet18
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes) 
    model.eval()  # Set model to evaluation mode


    # Define the attack method
    attack = torchattacks.FGSM(model, eps=0.03)  

    # Generate adversarial examples
    for images, labels in dataloader:
        images = images.requires_grad_(True)
        type1 = type(images)
        type2 = type(labels)
        # type3 = type(torch.tensor(labels))
        adv_imgs = attack(images, labels)
        adv_images.append(adv_imgs.detach())
        adv_labels.extend(labels)

    # Concatenate all adversarial images
    adv_images = torch.cat(adv_images)

    # Convert the adversarial images back to a format suitable for your DataFrame and combine them:

    # Convert adversarial images to PIL format
    adv_images_pil = [transforms.ToPILImage()(img) for img in adv_images]

    # Create a DataFrame for adversarial examples
    adv_df = pd.DataFrame({
        'image': adv_images_pil,
        'label': adv_labels
    })

    # Convert dataset back to DataFrame
    dataset_df = pd.DataFrame({
        'image': [dataset[i][0] for i in range(len(dataset))],  # Extract images
        'text': [dataset[i][1] for i in range(len(dataset))]    # Extract labels
    })

    # Combine with the original DataFrame
    augmented_df = pd.concat([dataset_df, adv_df], ignore_index=True)

    # Shuffle the combined DataFrame
    augmented_df = augmented_df.sample(frac=1).reset_index(drop=True)

    # If the combined DataFrame exceeds 1,000 samples, truncate it
    if len(augmented_df) > 1000:
        augmented_df = augmented_df[:1000]


    # 6. Save the Augmented Dataset 
    torch.save(augmented_df, "results/augmented_face_emojis.pt")
    return augmented_df



if __name__ == "__main__":
    # Load the original dataset
    df = pd.read_parquet("hf://datasets/valhalla/emoji-dataset/data/train-00000-of-00001-38cc4fa96c139e86.parquet")

    # Filter the dataset
    face_emojis = df[df["text"].str.contains("face", case=False, na=False)]

    # Adding another column to dataframe, of encoded labels. (to convert it to tensor type later.)
    label_encoder = LabelEncoder()
    # Fit encoder on unique labels and transform
    face_emojis["encoded_label"] = label_encoder.fit_transform(face_emojis["text"])

    # Augment the dataset
    augmented_df = augment_dataset(face_emojis)

    
    print("//////////////////////////////////////////////")
    print(augmented_df.info())
    # print(augmented_df.describe())
    # print(augmented_df["text"].unique())

