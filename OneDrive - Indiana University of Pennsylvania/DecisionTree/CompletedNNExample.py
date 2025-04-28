import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# 1. Load and preprocess the data
def load_and_preprocess_data(filepath):
    # Load the dataset
    df = pd.read_csv(filepath)
    
    # Define target and drop unnecessary columns
    y = df['Survived'].values
    
    # Drop unnecessary columns
    drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived']
    X = df.drop(columns=drop_cols)
    
    # Define numeric and categorical features
    numeric_features = ['Age', 'Fare', 'SibSp', 'Parch']
    categorical_features = ['Pclass', 'Sex', 'Embarked']
    
    # Create preprocessing pipelines for numeric and categorical features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine all transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Fit and transform the data
    X_processed = preprocessor.fit_transform(X)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, preprocessor

# 2. Create a custom dataset class
class TitanicDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 3. Create a neural network model
class myNet(nn.Module):
    def __init__(self, input_size):
        super(myNet, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.relu(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x

# 4. Training function
def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=100):
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            predictions = (outputs >= 0.5).float()
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # Evaluation
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                predictions = (outputs >= 0.5).float()
                total += labels.size(0)
                correct += (predictions == labels).sum().item()
        
        test_loss = running_loss / len(test_loader)
        test_acc = correct / total
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
    
    return model, train_losses, test_losses, train_accuracies, test_accuracies

# 5. Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs).squeeze()
            predictions = (outputs >= 0.5).float()
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
    
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy

# 6. Main function
def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, preprocessor = load_and_preprocess_data('titanic.csv')
    
    # Create datasets and dataloaders
    train_dataset = TitanicDataset(X_train, y_train)
    test_dataset = TitanicDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Create the model
    input_size = X_train.shape[1]
    model = myNet(input_size)
    print(f"Created model with input size: {input_size}")
    print(model)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    print("Training the model...")
    model, train_losses, test_losses, train_accuracies, test_accuracies = train_model(
        model, train_loader, test_loader, criterion, optimizer, num_epochs=100
    )
    
    # Evaluate the model
    print("Final evaluation:")
    final_accuracy = evaluate_model(model, test_loader)
    
    # Save the model
    torch.save(model.state_dict(), 'titanic_model.pth')
    print("Model saved as 'titanic_model.pth'")
    
    # Optional: Plot training curves
    try:
        import matplotlib.pyplot as plt
        
        # Plot loss curves
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss Curves')
        
        # Plot accuracy curves
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(test_accuracies, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy Curves')
        
        plt.tight_layout()
        plt.savefig('training_curves.png')
        print("Training curves saved as 'training_curves.png'")
    except ImportError:
        print("Matplotlib not available, skipping plot generation")

if __name__ == "__main__":
    main()