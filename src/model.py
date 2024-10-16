# import torch
# import torch.nn as nn
# import timm
# from sklearn.model_selection import train_test_split
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.preprocessing import LabelEncoder
# import pandas as pd

# # Swin Transformer Model Definition
# class SwinChurnNet(nn.Module):
#     def __init__(self):
#         super(SwinChurnNet, self).__init__()
#         self.swin_transformer = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=0)
#         self.fc1 = nn.Linear(768, 128)
#         self.fc2 = nn.Linear(128, 1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.swin_transformer(x)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = self.sigmoid(x)
#         return x

# # Data Loading (from preprocessed dataset)
# df = pd.read_csv('data/cleaned_telco_churn.csv')
# X = df.drop(columns=['Churn'])
# y = df['Churn']

# # Split data into training and validation sets
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# # Convert to tensors
# X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
# y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
# train_data = TensorDataset(X_train_tensor, y_train_tensor)
# train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# # Define the model, loss function, and optimizer
# model = SwinChurnNet()
# criterion = nn.BCELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# # Training loop
# def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
#     model.train()
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         for X_batch, y_batch in train_loader:
#             optimizer.zero_grad()
#             outputs = model(X_batch)
#             loss = criterion(outputs, y_batch)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")
#     print("Training complete.")

# train_model(model, train_loader, criterion, optimizer)

# # Save the trained model
# torch.save(model.state_dict(), 'swin_churn_model.pth')