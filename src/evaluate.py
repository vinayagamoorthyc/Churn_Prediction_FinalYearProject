# import torch
# import pandas as pd
# from sklearn.model_selection import train_test_split

# # Load the model
# model = SwinChurnNet()
# model.load_state_dict(torch.load('swin_churn_model.pth'))
# model.eval()

# # Load validation data
# df = pd.read_csv('data/cleaned_telco_churn.csv')
# X = df.drop(columns=['Churn'])
# y = df['Churn']

# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# # Convert validation data to tensors
# X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
# y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

# # Evaluate the model on validation data
# with torch.no_grad():
#     outputs = model(X_val_tensor)
#     predictions = (outputs > 0.5).float()
#     accuracy = (predictions == y_val_tensor).float().mean()
#     print(f"Validation Accuracy: {accuracy.item() * 100:.2f}%")