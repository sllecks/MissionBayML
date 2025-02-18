import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump, load
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

class UmpireModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.model_filename = 'ump_model.joblib'
        self.scaler_filename = 'ump_scaler.joblib'
        self.features = [
            'plate_x', 'plate_z',  # pitch location
            'release_speed',       # velocity
            'pfx_x', 'pfx_z',     # movement
            'release_pos_x', 'release_pos_z'  # release position
        ]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def prepare_data(self, data):
        """Prepare features and labels for the umpire model."""
        # Filter for non-swing pitches
        mask = data['description'].isin(['ball', 'called_strike', 'hit_by_pitch'])
        filtered_data = data[mask]
        
        # Create features
        X = filtered_data[self.features].copy()
        X = X.fillna(0)  # Handle missing values
        X_scaled = self.scaler.fit_transform(X)
        
        # Create labels
        conditions = [
            (filtered_data['description'] == 'ball'),
            (filtered_data['description'] == 'called_strike'),
            (filtered_data['description'] == 'hit_by_pitch')
        ]
        choices = [0, 1, 2]
        y = np.select(conditions, choices, default=-1)
        
        return X_scaled, y

    def _evaluate_model(self, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred):
        """Evaluate model performance and save metrics."""
        # Get performance metrics
        train_report = classification_report(y_train, y_train_pred, 
            target_names=['Ball', 'Called Strike', 'Hit By Pitch'], output_dict=True)
        test_report = classification_report(y_test, y_test_pred, 
            target_names=['Ball', 'Called Strike', 'Hit By Pitch'], output_dict=True)
        
        # Create performance record
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        performance_data = {
            'timestamp': timestamp,
            'train_accuracy': train_report['accuracy'],
            'train_weighted_f1': train_report['weighted avg']['f1-score'],
            'test_accuracy': test_report['accuracy'],
            'test_weighted_f1': test_report['weighted avg']['f1-score']
        }
        
        # Save performance metrics to CSV
        performance_file = os.path.join('performance', 'ump_model_performance.csv')
        performance_df = pd.DataFrame([performance_data])
        
        if os.path.exists(performance_file):
            performance_df.to_csv(performance_file, mode='a', header=False, index=False)
        else:
            performance_df.to_csv(performance_file, index=False)
        
        # Create and save confusion matrix plot
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_test_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Umpire Model Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plots_dir = 'plots'
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        plt.savefig(os.path.join(plots_dir, 'ump_confusion_matrix.png'))
        plt.close()

    def train(self, X, y):
        """Train the umpire model using default parameters"""
        default_params = {
            'epochs': 50,
            'batch_size': 64,
            'learning_rate': 0.001,
            'hidden_size': 64,
            'dropout_rate': 0.2
        }
        return self._train_model(X, y, **default_params)

    def _train_model(self, X, y, epochs=50, batch_size=64, learning_rate=0.001,
                    hidden_size=64, dropout_rate=0.2):
        """Core training logic for the umpire model"""
        # Convert to tensors and move to device
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_tensor.cpu().numpy(), y_tensor.cpu().numpy(), test_size=0.2, random_state=42
        )
        
        # Move data to device
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.LongTensor(y_train).to(self.device)
        X_test = torch.FloatTensor(X_test).to(self.device)
        y_test = torch.LongTensor(y_test).to(self.device)
        
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model with wider layers
        self.model = nn.Sequential(
            nn.Linear(len(self.features), hidden_size * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size * 2),  # Add batch normalization
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),  # Add batch normalization
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 3),
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)  # Add weight decay
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        # Training loop with loss tracking
        best_loss = float('inf')
        patience_counter = 0
        patience_limit = 10
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            scheduler.step(avg_loss)
            
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience_limit:
                print(f'Early stopping at epoch {epoch+1}')
                break
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
        
        # Evaluate model
        self.model.eval()
        with torch.no_grad():
            y_train_pred = torch.argmax(self.model(X_train), dim=1).cpu().numpy()
            y_test_pred = torch.argmax(self.model(X_test), dim=1).cpu().numpy()
            
        self._evaluate_model(
            X_train.cpu().numpy(), X_test.cpu().numpy(),
            y_train.cpu().numpy(), y_test.cpu().numpy(),
            y_train_pred, y_test_pred
        )
        
        return self.model

    def save_model(self):
        """Save model and scaler to files."""
        models_dir = 'models'
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(models_dir, f'ump_model_{timestamp}.joblib')
        scaler_path = os.path.join(models_dir, f'ump_scaler_{timestamp}.joblib')
        
        dump(self.model, model_path)
        dump(self.scaler, scaler_path)
        print(f"Model saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")

    def load_model(self):
        """Load a pre-trained model and scaler from files."""
        models_dir = 'models'
        model_path = os.path.join(models_dir, self.model_filename)
        scaler_path = os.path.join(models_dir, self.scaler_filename)
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Model or scaler file not found in {models_dir}")
        
        self.model = load(model_path)
        self.scaler = load(scaler_path)
        print(f"Loaded model from: {model_path}")
        print(f"Loaded scaler from: {scaler_path}")

    def predict(self, X):
        """Predict pitch outcomes"""
        if not isinstance(X, torch.Tensor):
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        else:
            X_tensor = X.to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        
        # Add debug print
        print(f"Raw model outputs: {outputs}")
        print(f"Predictions shape: {predictions.shape}")
        print(f"Unique predictions: {np.unique(predictions)}")
        
        return predictions

    def predict_proba(self, X):
        """Predict probabilities for each outcome"""
        if not isinstance(X, torch.Tensor):
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        else:
            X_tensor = X.to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
        
        # Add debug print
        print(f"Probabilities shape: {probabilities.shape}")
        print(f"Sample probabilities:\n{probabilities}")
        
        return probabilities

def main():
    # Load the data
    try:
        data = pd.read_csv("pitch_data_2023.csv")
        
        # Initialize model
        ump_model = UmpireModel()
        
        # Try to load existing model, train new one if not found
        try:
            ump_model.load_model()
            print("Model loaded successfully")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Training new model instead...")
            X, y = ump_model.prepare_data(data)
            print(f"Training data shape: {X.shape}")
            print(f"Unique labels in training: {np.unique(y)}")
            ump_model.train(X, y)
            ump_model.save_model()
            
        # Example prediction with proper data handling
        mask = data['description'].isin(['ball', 'called_strike', 'hit_by_pitch'])
        filtered_data = data[mask]
        
        # Ensure features are numeric and handle any missing values
        sample_pitch = filtered_data[ump_model.features].head(5).fillna(0)
        sample_events = filtered_data['description'].head(5)
        
        print("\nSample input features:")
        print(sample_pitch)
        print("\nActual events:")
        print(sample_events)
        
        # Make predictions
        print("\nMaking predictions...")
        predictions = ump_model.predict(sample_pitch)
        probabilities = ump_model.predict_proba(sample_pitch)
        
        # Print results with better formatting
        outcome_map = {0: "Ball", 1: "Called Strike", 2: "Hit By Pitch"}
        print("\nSample Predictions:")
        for i, (pred, prob, actual) in enumerate(zip(predictions, probabilities, sample_events)):
            print(f"\nPitch {i+1}:")
            print(f"Actual: {actual.replace('_', ' ').title()}")
            print(f"Predicted: {outcome_map[pred]}")
            print("Probabilities:")
            print(f"  Ball: {prob[0]:.3f}")
            print(f"  Strike: {prob[1]:.3f}")
            print(f"  HBP: {prob[2]:.3f}")
            
    except FileNotFoundError:
        print("Error: pitch_data_2023.csv not found. Please ensure the data file exists.")

if __name__ == "__main__":
    main()
