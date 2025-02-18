import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from joblib import dump, load
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from itertools import product

def create_swing_labels(df):
    # Define what constitutes a swing
    swing_descriptions = ['swinging_strike', 'hit_into_play', 'foul', 'foul_tip']
    
    # Create binary label: 1 for swing, 0 for no swing
    df['swing'] = df['description'].isin(swing_descriptions).astype(int)
    
    return df

class SwingPredictor:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.model_filename = 'swing_model.joblib'
        self.scaler_filename = 'swing_scaler.joblib'
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
        y = df['swing']
        
        # Handle missing values
        X = X.fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def train(self, X, y):
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize and train the model
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        
        # Fit the model
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        self._evaluate_model(X_train, X_test, y_train, y_test)
        
        return self.model
    
    def train_neural_network(self, X, y, grid_search=True):
        """
        Train the model with optional grid search for hyperparameters
        Args:
            X: Features
            y: Target labels
            grid_search: Whether to perform grid search for hyperparameters
        """
        if grid_search:
            # Define parameter grid
            param_grid = {
                'epochs': [30, 50, 100],
                'batch_size': [32, 64, 128],
                'learning_rate': [0.01, 0.001, 0.0001],
                'hidden_size': [32, 64, 128],
                'dropout_rate': [0.1, 0.2, 0.3]
            }
            
            best_params = self._grid_search(X, y, param_grid)
            return self._train_model(X, y, **best_params)
        else:
            # Use default parameters
            default_params = {
                'epochs': 50,
                'batch_size': 64,
                'learning_rate': 0.001,
                'hidden_size': 64,
                'dropout_rate': 0.2
            }
            return self._train_model(X, y, **default_params)

    def _grid_search(self, X, y, param_grid):
        """Perform grid search to find best hyperparameters"""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y.values).reshape(-1, 1)
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_tensor, y_tensor, test_size=0.2, random_state=42
        )
        
        best_val_loss = float('inf')
        best_params = None
        
        # Generate all combinations of parameters
        param_combinations = [dict(zip(param_grid.keys(), v)) 
                            for v in product(*param_grid.values())]
        
        for params in param_combinations:
            print(f"\nTrying parameters: {params}")
            
            # Train model with current parameters
            model = self._train_model(
                X_train.numpy(), y_train.numpy(), 
                validation_data=(X_val.numpy(), y_val.numpy()),
                **params
            )
            
            # Evaluate on validation set
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val.to(self.device))
                val_loss = nn.BCELoss()(val_outputs, y_val.to(self.device))
            
            print(f"Validation loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = params
                
        print(f"\nBest parameters found: {best_params}")
        print(f"Best validation loss: {best_val_loss:.4f}")
        
        return best_params

    def _train_model(self, X, y, epochs=50, batch_size=64, learning_rate=0.001,
                    hidden_size=64, dropout_rate=0.2, validation_data=None):
        """Core training logic, extracted from original train method"""
        # Convert data to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1)
        
        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X_tensor.numpy(), y_tensor.numpy(), test_size=0.2, random_state=42
        )
        
        # Create data loader
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Define the neural network with configurable parameters
        self.model = nn.Sequential(
            nn.Linear(len(self.features), hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size//2, 1),
            nn.Sigmoid()
        ).to(self.device)
        
        # Rest of the training logic remains the same as original train method
        criterion = nn.BCELoss()
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
        
        # After training, evaluate and record performance
        self.model.eval()
        with torch.no_grad():
            y_train_pred = (self.model(torch.FloatTensor(X_train).to(self.device)) > 0.5).float().cpu().numpy()
            y_test_pred = (self.model(torch.FloatTensor(X_test).to(self.device)) > 0.5).float().cpu().numpy()
            
        self._evaluate_model(X_train, X_test, y_train, y_test, y_train_pred, y_test_pred)
        
        return self.model

    def _evaluate_model(self, X_train, X_test, y_train, y_test, y_train_pred=None, y_test_pred=None):
        # If predictions aren't provided, generate them (for RandomForest)
        if y_train_pred is None or y_test_pred is None:
            y_train_pred = self.model.predict(X_train)
            y_test_pred = self.model.predict(X_test)
        
        # Get performance metrics
        train_report = classification_report(y_train, y_train_pred, output_dict=True)
        test_report = classification_report(y_test, y_test_pred, output_dict=True)
        
        # Create performance record
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        performance_data = {
            'timestamp': timestamp,
            'train_accuracy': train_report['accuracy'],
            'train_precision': train_report['weighted avg']['precision'],
            'train_recall': train_report['weighted avg']['recall'],
            'train_f1': train_report['weighted avg']['f1-score'],
            'test_accuracy': test_report['accuracy'],
            'test_precision': test_report['weighted avg']['precision'],
            'test_recall': test_report['weighted avg']['recall'],
            'test_f1': test_report['weighted avg']['f1-score']
        }
        
        # Save performance to CSV
        performance_dir = 'performance'
        if not os.path.exists(performance_dir):
            os.makedirs(performance_dir)
        
        performance_file = os.path.join(performance_dir, 'model_performance.csv')
        performance_df = pd.DataFrame([performance_data])
        
        if os.path.exists(performance_file):
            performance_df.to_csv(performance_file, mode='a', header=False, index=False)
        else:
            performance_df.to_csv(performance_file, index=False)
        
        # Create confusion matrix plot
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_test_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot
        plots_dir = 'plots'
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        plt.savefig(os.path.join(plots_dir, 'swing_confusion_matrix.png'))
        plt.close()
        
        # Feature importance plot - only for Random Forest
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(12, 6))
            sns.barplot(x='importance', y='feature', data=feature_importance)
            plt.title('Feature Importance for Swing Prediction')
            plt.savefig(os.path.join(plots_dir, 'swing_feature_importance.png'))
            plt.close()
    
    def save_model(self):
        # Create models directory if it doesn't exist
        models_dir = 'models'
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        # Save model and scaler with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(models_dir, f'swing_model_{timestamp}.joblib')
        scaler_path = os.path.join(models_dir, f'swing_scaler_{timestamp}.joblib')
        
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

def main():
    # Load the data
    df = pd.read_csv('pitch_data_2023.csv')
    
    # Create swing labels
    df = create_swing_labels(df)
    
    # Initialize predictor
    predictor = SwingPredictor()
    
    # Try to load existing model, train new one if not found
    try:
        predictor.load_model()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Training new model instead...")
        X, y = predictor.prepare_data(df)
        predictor.train(X, y)
        predictor.save_model()

if __name__ == "__main__":
    main()