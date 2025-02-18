import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from joblib import dump, load
from itertools import product

def create_contact_labels(df):
    """
    Create labels for contact outcomes when a swing occurs:
    0: swinging_strike
    1: foul_ball (includes foul_tip)
    2: in_play
    """
    # Filter for swings only
    swing_mask = df['description'].isin(['swinging_strike', 'hit_into_play', 'foul', 'foul_tip'])
    df = df[swing_mask].copy()
    
    # Create categorical label
    conditions = [
        df['description'] == 'swinging_strike',
        df['description'].isin(['foul', 'foul_tip']),
        df['description'] == 'hit_into_play'
    ]
    choices = [0, 1, 2]
    
    df['contact_result'] = np.select(conditions, choices, default=-1)
    return df

class ContactPredictor:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.model_filename = 'contact_model.joblib'
        self.scaler_filename = 'contact_scaler.joblib'
        self.features = [
            'release_speed', 'release_spin_rate', 'pfx_x', 'pfx_z',
            'plate_x', 'plate_z', 'balls', 'strikes',
            'effective_speed', 'release_pos_x', 'release_pos_z',
            'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az',
            'release_extension', 'arm_angle', 'spin_axis',
            'bat_speed', 'swing_length', 'pitch_number'
        ]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def prepare_data(self, df):
        # Create features and labels
        X = df[self.features].copy()
        y = df['contact_result']
        
        # Handle missing values
        X = X.fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def train(self, X, y):
        """
        Train the model with default parameters
        """
        default_params = {
            'epochs': 200,
            'batch_size': 64,
            'learning_rate': 0.001,
            'hidden_size': 64,
            'dropout_rate': 0.2
        }
        return self._train_model(X, y, **default_params)

    def _train_model(self, X, y, epochs=50, batch_size=64, learning_rate=0.001,
                    hidden_size=64, dropout_rate=0.2):
        """Core training logic for the contact model"""
        # Convert pandas Series to numpy array before creating tensor
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y.values)  # Add .values here
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_tensor.numpy(), y_tensor.numpy(), test_size=0.2, random_state=42
        )
        
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Define the neural network for 3-class classification
        self.model = nn.Sequential(
            nn.Linear(len(self.features), hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size//2, 3),  # 3 classes: swinging strike, foul, in play
        ).to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')
        
        # Evaluate model
        self.model.eval()
        with torch.no_grad():
            y_train_pred = torch.argmax(self.model(torch.FloatTensor(X_train).to(self.device)), dim=1).cpu().numpy()
            y_test_pred = torch.argmax(self.model(torch.FloatTensor(X_test).to(self.device)), dim=1).cpu().numpy()
            
        self._evaluate_model(X_train, X_test, y_train, y_test, y_train_pred, y_test_pred)
        
        return self.model

    def _evaluate_model(self, X_train, X_test, y_train, y_test, y_train_pred, y_test_pred):
        # Get performance metrics
        train_report = classification_report(y_train, y_train_pred, output_dict=True)
        test_report = classification_report(y_test, y_test_pred, output_dict=True)
        
        # Create performance record
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        performance_data = {
            'timestamp': timestamp,
            'train_accuracy': train_report['accuracy'],
            'train_weighted_f1': train_report['weighted avg']['f1-score'],
            'test_accuracy': test_report['accuracy'],
            'test_weighted_f1': test_report['weighted avg']['f1-score']
        }
        
        # Save performance metrics
        performance_dir = 'performance'
        if not os.path.exists(performance_dir):
            os.makedirs(performance_dir)
        
        performance_file = os.path.join(performance_dir, 'contact_model_performance.csv')
        performance_df = pd.DataFrame([performance_data])
        
        if os.path.exists(performance_file):
            performance_df.to_csv(performance_file, mode='a', header=False, index=False)
        else:
            performance_df.to_csv(performance_file, index=False)
        
        # Create confusion matrix plot
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_test_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Contact Prediction Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plots_dir = 'plots'
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        plt.savefig(os.path.join(plots_dir, 'contact_confusion_matrix.png'))
        plt.close()

    def save_model(self):
        models_dir = 'models'
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(models_dir, f'contact_model_{timestamp}.joblib')
        scaler_path = os.path.join(models_dir, f'contact_scaler_{timestamp}.joblib')
        
        dump(self.model, model_path)
        dump(self.scaler, scaler_path)
        print(f"Model saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")

    def load_model(self):
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
        """Predict contact outcomes"""
        if not isinstance(X, torch.Tensor):
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        else:
            X_tensor = X.to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
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
        return probabilities

def main():
    # Load the data
    df = pd.read_csv('pitch_data_2023.csv')
    print(f"Total rows in original dataset: {len(df)}")
    
    # Create contact labels
    df = create_contact_labels(df)
    print(f"Rows after filtering for swings only: {len(df)}")
    print("\nBreakdown by contact type:")
    print(df['contact_result'].value_counts())
    
    # Initialize predictor
    predictor = ContactPredictor()
    
    # Try to load existing model, train new one if not found
    try:
        predictor.load_model()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Training new model instead...")
        X, y = predictor.prepare_data(df)
        predictor.train(X, y)
        predictor.save_model()

    # Show sample predictions
    print("\nGenerating sample predictions...")
    sample_swings = df.head(10)
    sample_features = sample_swings[predictor.features].fillna(0)
    actual_results = sample_swings['contact_result']

    predictions = predictor.predict(sample_features)
    probabilities = predictor.predict_proba(sample_features)

    # Print results with formatting
    outcome_map = {0: "Swinging Strike", 1: "Foul Ball", 2: "In Play"}
    print("\nSample Predictions:")
    for i, (pred, prob, actual) in enumerate(zip(predictions, probabilities, actual_results)):
        print(f"\nSwing {i+1}:")
        print(f"Actual: {outcome_map[actual]}")
        print(f"Predicted: {outcome_map[pred]}")
        print("Probabilities:")
        print(f"  Swinging Strike: {prob[0]:.3f}")
        print(f"  Foul Ball: {prob[1]:.3f}")
        print(f"  In Play: {prob[2]:.3f}")

if __name__ == "__main__":
    main()
