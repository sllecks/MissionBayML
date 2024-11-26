# !pip3 install scikit-learn pandas numpy matplotlib seaborn xgboost lightgbm imbalanced-learn tensorflow tabulate
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import precision_recall_curve, average_precision_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
import tensorflow as tf
from joblib import dump, load

# Load the data
def load_data():
    train_df = pd.read_csv('train.csv')
    return train_df

def create_features(df):
    # First, ensure the dataframe is properly sorted for rolling calculations
    df = df.sort_values(by=['uid', 'pitch_number']).copy()  # Add .copy() to avoid SettingWithCopyWarning
    
    # Basic pitch movement features
    print("Creating basic pitch features...")
    df['total_break'] = np.sqrt(df['pfx_x']**2 + df['pfx_z']**2)
    df['break_angle'] = np.arctan2(df['pfx_z'], df['pfx_x'])
    
    # Game situation features
    print("Creating game situation features...")
    df['opposite_hand'] = df['is_lhp'] != df['is_lhb']
    # Handle potential NaN values in base runners
    df['on_1b'] = df['on_1b'].fillna(0)
    df['on_2b'] = df['on_2b'].fillna(0)
    df['on_3b'] = df['on_3b'].fillna(0)
    df['on_base'] = df['on_1b'] + df['on_2b'] + df['on_3b']
    df['late_game'] = df['inning'] >= 7
    df['pressure'] = (df['outs_when_up'] == 2) & ((df['on_2b'] == 1) | (df['on_3b'] == 1))
    df['clutch'] = df['pressure'] & df['late_game']
    
    # Pitcher state features
    print("Creating pitcher state features...")
    df['fatigue'] = df['pitch_number'].fillna(0) / 110  # Handle missing pitch numbers
    
    # Advanced physics features
    print("Creating advanced physics features...")
    df['momentum_x'] = df['release_speed'] * df['release_pos_x']
    df['momentum_z'] = df['release_speed'] * df['release_pos_z']
    df['total_acceleration'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
    df['pitch_break_horizontal'] = df['vx0'] - df['pfx_x']
    df['pitch_break_vertical'] = df['vz0'] - df['pfx_z']
    
    # Rolling statistics with proper grouping and handling of edge cases
    print("Creating rolling statistics...")
    # Initialize rolling stat columns with zeros
    rolling_columns = ['avg_speed_last_5', 'avg_speed_last_10', 
                      'avg_spin_last_5', 'avg_spin_last_10']
    for col in rolling_columns:
        df[col] = 0.0
    
    # Calculate rolling stats for each pitcher (uid)
    for uid in df['uid'].unique():
        mask = df['uid'] == uid
        if mask.sum() > 0:  # Only process if there are pitches for this uid
            # Speed rolling averages
            df.loc[mask, 'avg_speed_last_5'] = (
                df.loc[mask, 'release_speed']
                .rolling(window=5, min_periods=1)
                .mean()
                .fillna(df.loc[mask, 'release_speed'])
            )
            df.loc[mask, 'avg_speed_last_10'] = (
                df.loc[mask, 'release_speed']
                .rolling(window=10, min_periods=1)
                .mean()
                .fillna(df.loc[mask, 'release_speed'])
            )
            
            # Spin rolling averages
            df.loc[mask, 'avg_spin_last_5'] = (
                df.loc[mask, 'release_spin_rate']
                .rolling(window=5, min_periods=1)
                .mean()
                .fillna(df.loc[mask, 'release_spin_rate'])
            )
            df.loc[mask, 'avg_spin_last_10'] = (
                df.loc[mask, 'release_spin_rate']
                .rolling(window=10, min_periods=1)
                .mean()
                .fillna(df.loc[mask, 'release_spin_rate'])
            )
    
    # Add interaction features
    print("Creating interaction features...")
    # Handle potential division by zero in speed_spin_ratio
    df['speed_spin_ratio'] = np.where(
        df['release_spin_rate'] > 0,
        df['release_speed'] / df['release_spin_rate'],
        0
    )
    df['vertical_approach_angle'] = np.arctan2(df['vz0'], df['vx0'])
    
    # Add count features
    print("Creating count features...")
    df['favorable_count'] = ((df['balls'] > df['strikes']) | 
                           ((df['balls'] == 3) & (df['strikes'] < 3)))
    
    # Add more advanced rolling stats
    print("Creating advanced rolling stats...")
    # Calculate spin efficiency safely
    max_spin_by_uid = df.groupby('uid')['release_spin_rate'].transform('max')
    df['spin_efficiency'] = np.where(
        max_spin_by_uid > 0,
        df['release_spin_rate'] / max_spin_by_uid,
        0
    )
    
    # Calculate pitch tunneling
    df['pitch_tunneling'] = (
        df.groupby('uid')['release_pos_x'].diff().abs() +
        df.groupby('uid')['release_pos_z'].diff().abs()
    ).fillna(0)
    
    # Fill any remaining NaN values
    print("Filling any remaining NaN values...")
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(0)
    
    # Verify all features were created
    expected_features = [
        'total_break', 'break_angle', 'opposite_hand', 'on_base',
        'late_game', 'pressure', 'clutch', 'fatigue', 'momentum_x',
        'momentum_z', 'total_acceleration', 'pitch_break_horizontal',
        'pitch_break_vertical', 'avg_speed_last_5', 'avg_speed_last_10',
        'avg_spin_last_5', 'avg_spin_last_10', 'speed_spin_ratio',
        'vertical_approach_angle', 'favorable_count', 'spin_efficiency',
        'pitch_tunneling'
    ]
    
    missing_features = set(expected_features) - set(df.columns)
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
    
    print("Feature creation complete.")
    return df

def prepare_data(df):
    # Define feature groups
    pitch_features = [
        'release_speed', 'release_spin_rate', 'pfx_x', 'pfx_z',
        'plate_x', 'plate_z', 'total_break', 'break_angle',
        'pitch_type', 'effective_speed', 'momentum_x', 'momentum_z',
        'total_acceleration', 'pitch_break_horizontal', 'pitch_break_vertical'
    ]
    
    batter_features = [
        'is_lhb', 'bat_speed', 'swing_length', 'spray_angle'
    ]
    
    pitcher_features = [
        'is_lhp', 'fatigue'
    ]
    
    game_situation_features = [
        'opposite_hand', 'on_base', 'late_game',
        'pressure', 'clutch'
    ]
    
    rolling_stats_features = [
        'avg_speed_last_5', 'avg_speed_last_10',
        'avg_spin_last_5', 'avg_spin_last_10'
    ]
    
    # Store all features as a class attribute
    all_features = (pitch_features + batter_features + pitcher_features + 
                   game_situation_features + rolling_stats_features)
    
    # Remove duplicates while preserving order
    all_features = list(dict.fromkeys(all_features))
    
    # Create feature groups dictionary
    feature_groups = {
        'pitch': pitch_features,
        'batter': batter_features,
        'pitcher': pitcher_features,
        'game_situation': game_situation_features,
        'rolling_stats': rolling_stats_features
    }
    
    # Create target variable
    hit_outcomes = ['out','single', 'double', 'triple', 'home_run']
    df['hit_type'] = df['outcome']
    
    # Before scaling, separate numeric and categorical features
    categorical_features = ['pitch_type', 'pitch_name']  # Add any other categorical columns
    numeric_features = [f for f in all_features if f not in categorical_features]
    
    # Encode categorical features
    le_dict = {}
    for cat_feature in categorical_features:
        if cat_feature in all_features:
            le = LabelEncoder()
            df[f'{cat_feature}_encoded'] = le.fit_transform(df[cat_feature])
            le_dict[cat_feature] = le
            # Replace the categorical feature with its encoded version in the features list
            all_features[all_features.index(cat_feature)] = f'{cat_feature}_encoded'
    
    # Now scale only numeric features
    scaler = StandardScaler()
    X = df[all_features].copy()  # Use all features (now with encoded categoricals)
    X = scaler.fit_transform(X)
    y = df['hit_type']

    # Save the scaler
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    scaler_path = os.path.join(models_dir, f'scaler_{timestamp}.joblib')
    dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")

    # Encode the target variable before SMOTE
    le = LabelEncoder()
    y = le.fit_transform(y)
    le_dict['hit_type'] = le

    # SMOTE for handling class imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Feature selection using Random Forest instead of Lasso
    selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
    X_selected = selector.fit_transform(X_resampled, y_resampled)
    selected_features = [all_features[i] for i in range(len(all_features)) 
                        if selector.get_support()[i]]

    # Calculate class weights
    class_weights = dict(zip(
        np.unique(y_resampled),
        len(y_resampled) / (len(np.unique(y_resampled)) * np.bincount(y_resampled))
    ))

    return X_selected, y_resampled, selected_features, le_dict['hit_type'], scaler, feature_groups, class_weights

def train_model(X_train, y_train, X_test, y_test, features, feature_groups=None, class_weights=None, scaler=None):

    
    print("\n========== MODEL TRAINING DETAILS ==========")
    
    # Print dataset sizes
    print("\nDataset Sizes:")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    print(f"Number of features: {X_train.shape[1]}")
    
    # Print class distribution
    print("\nClass Distribution:")
    train_dist = Counter(y_train)
    test_dist = Counter(y_test)
    print("\nTraining set:")
    for label, count in train_dist.items():
        percentage = count/len(y_train)*100
        print(f"{label}: {count} samples ({percentage:.2f}%)")
    print("\nTest set:")
    for label, count in test_dist.items():
        percentage = count/len(y_test)*100
        print(f"{label}: {count} samples ({percentage:.2f}%)")
    
    # Calculate class weights
    class_weights = dict(zip(
        np.unique(y_train),
        len(y_train) / (len(np.unique(y_train)) * np.bincount(y_train))
    ))
    print("\nClass Weights:")
    for label, weight in class_weights.items():
        print(f"{label}: {weight:.4f}")
    
    print("\n========== TRAINING MODEL ==========")
    # Initialize model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight=class_weights,
        verbose=1  # Add verbosity
    )
    
    print("\nFitting model...")
    model.fit(X_train, y_train)
    
    # Make predictions
    print("\nMaking predictions...")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_train_proba = model.predict_proba(X_train)
    y_test_proba = model.predict_proba(X_test)
    
    print("\n========== MODEL PERFORMANCE ==========")
    
    # Print basic metrics
    print("\nAccuracy Scores:")
    print(f"Training Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    
    # Detailed classification reports
    print("\nTraining Classification Report:")
    print(classification_report(y_train, y_train_pred, zero_division=0))
    print("\nTest Classification Report:")
    print(classification_report(y_test, y_test_pred, zero_division=0))
    
    # Create plots directory if it doesn't exist
    plots_dir = 'plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Update confusion matrix plot
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'))
    plt.close()
    
    print("\n========== FEATURE IMPORTANCE ==========")
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    # Ensure feature importances match feature count
    feature_importance = feature_importance.iloc[:len(features)]
    
    print("\nTop 15 Most Important Features:")
    print(feature_importance.head(15))
    
    # Update feature importance plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
    plt.title('Top 15 Feature Importance')
    plt.savefig(os.path.join(plots_dir, 'feature_importance.png'))
    plt.close()
    
    # Group importance
    if feature_groups:
        print("\nFeature Importance by Group:")
        group_importance = {}
        for group_name, group_features in feature_groups.items():
            group_imp = feature_importance[
                feature_importance['feature'].isin(group_features)
            ]['importance'].sum()
            group_importance[group_name] = group_imp
            print(f"{group_name}: {group_imp:.4f}")
        
        # Update group importance plot
        plt.figure(figsize=(10, 6))
        group_imp_df = pd.DataFrame(list(group_importance.items()), 
                                  columns=['Group', 'Importance'])
        sns.barplot(x='Importance', y='Group', data=group_imp_df)
        plt.title('Feature Group Importance')
        plt.savefig(os.path.join(plots_dir, 'group_importance.png'))
        plt.close()
    
    print("\n========== MODEL PARAMETERS ==========")
    print("\nRandom Forest Parameters:")
    for param, value in model.get_params().items():
        print(f"{param}: {value}")
    
    print("\n========== CROSS-VALIDATION ==========")
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print("\n5-Fold Cross-validation Scores:")
    print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"Individual Fold Scores: {cv_scores}")
    
    # Define hyperparameter search space
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Initialize base model
    base_model = RandomForestClassifier(random_state=42, class_weight=class_weights)
    
    # Perform randomized search
    random_search = RandomizedSearchCV(
        base_model, 
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=2
    )
    
    print("\nPerforming hyperparameter tuning...")
    random_search.fit(X_train, y_train)
    
    # Use best model
    model = random_search.best_estimator_
    print(f"\nBest parameters found: {random_search.best_params_}")
    
    # Update Precision-Recall curve plot
    plt.figure(figsize=(10, 8))
    for i in range(len(model.classes_)):
        precision, recall, _ = precision_recall_curve(
            y_test == i,
            y_test_proba[:, i]
        )
        avg_precision = average_precision_score(y_test == i, y_test_proba[:, i])
        plt.plot(recall, precision, 
                label=f'Class {i} (AP = {avg_precision:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Each Class')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'precision_recall_curve.png'))
    plt.close()
    
    print("\n========== SAVING MODEL ==========")
    
    # Create models directory if it doesn't exist
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Add timestamp to model files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save the model and its components
    model_path = os.path.join(models_dir, f'random_forest_model_{timestamp}.joblib')
    dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save feature names
    feature_path = os.path.join(models_dir, f'features_{timestamp}.joblib')
    dump(features, feature_path)
    print(f"Features saved to: {feature_path}")
    
    # Only save scaler if it was provided
    if scaler is not None:
        scaler_path = os.path.join(models_dir, f'scaler_{timestamp}.joblib')
        dump(scaler, scaler_path)
        print(f"Scaler saved to: {scaler_path}")
    
    return model

def predict_and_save(model, X_test, test_df_uids, output_file):
    # Find the latest model file
    models_dir = 'models'
    model_files = [f for f in os.listdir(models_dir) if f.startswith('random_forest_model_')]
    if model_files:
        latest_model_file = max(model_files)  # Gets the most recent timestamp
        model_path = os.path.join(models_dir, latest_model_file)
        print(f"\nLoading latest model from: {model_path}")
        model = load(model_path)
    else:
        print("\nNo saved model found, using provided model")

    probabilities = model.predict_proba(X_test)
    
    # Create predictions directory if it doesn't exist
    predictions_dir = 'predictions'
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)
    
    # Add timestamp to filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'predictions_{timestamp}.csv'
    output_path = os.path.join(predictions_dir, filename)
    
    # Create DataFrame with predictions
    predictions_df = pd.DataFrame({
        'uid': test_df_uids,
        'out': probabilities[:, 0],
        'single': probabilities[:, 1],
        'double': probabilities[:, 2],
        'triple': probabilities[:, 3],
        'home_run': probabilities[:, 4]
    })
    
    # Normalize probabilities
    prediction_columns = ['out', 'single', 'double', 'triple', 'home_run']
    row_sums = predictions_df[prediction_columns].sum(axis=1)
    for col in prediction_columns:
        predictions_df[col] = predictions_df[col] / row_sums
    
    # Save predictions
    predictions_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")
    return predictions_df

def create_ensemble_model():
    # Define base models
    models = [
        ('rf', RandomForestClassifier(random_state=42)),
        ('xgb', XGBClassifier(random_state=42)),
        ('lgbm', LGBMClassifier(random_state=42))
    ]
    
    # Create stacking classifier
    stacking = StackingClassifier(
        estimators=models,
        final_estimator=LogisticRegression(),
        cv=5
    )
    
    return stacking

def main():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # Assuming you have at least one GPU:
    gpu_device = tf.config.list_physical_devices('GPU')[0]
    tf.config.experimental.set_memory_growth(gpu_device, True) 
    
    # Fix: Add quotes around file paths
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    sample_submission_df = pd.read_csv('sample_submission.csv')
    
    # Display the first 5 rows of each DataFrame
    print("First 5 rows of train_df:")
    print(train_df.head().to_markdown(index=False, numalign="left", stralign="left"))

    print("\nFirst 5 rows of test_df:")
    print(test_df.head().to_markdown(index=False, numalign="left", stralign="left"))

    print("\nFirst 5 rows of sample_submission_df:")
    print(sample_submission_df.head().to_markdown(index=False, numalign="left", stralign="left"))

    # Print the column names and their data types for each DataFrame
    print("\nColumn names and their data types for train_df:")
    print(train_df.info())

    print("\nColumn names and their data types for test_df:")
    print(test_df.info())

    print("\nColumn names and their data types for sample_submission_df:")
    print(sample_submission_df.info())
    print("Loading and preparing training data...")
    df = load_data()
    df = create_features(df)
    X, y, features, label_encoder, scaler, feature_groups, class_weights = prepare_data(df)
    
    # Split the data
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train the model
    print("\nTraining model...")
    model = train_model(X_train, y_train, X_test, y_test, features, feature_groups, class_weights, scaler)
    
    # Load and process test data
    print("\nLoading test data...")
    test_df = pd.read_csv('test.csv')
    test_df = create_features(test_df)  # Create features for test data
    
    # Process test data using the same preparation steps
    X_test_final = scaler.transform(test_df[features])  # Use the same features and scaler from training
    
    try:
        # Make predictions using the properly processed test data
        print("\nMaking predictions...")
        predictions = predict_and_save(model, X_test_final, test_df['uid'], 'predictions.csv')
        
        # Print feature importance
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 most important features:")
        print(feature_importance.head(10))
        
        # Print the mapping of encoded labels
        print("\nLabel Encoding Mapping:")
        for i, label in enumerate(label_encoder.classes_):
            print(f"{i}: {label}")
            
    except Exception as e:
        print(f"\nError during prediction: {str(e)}")
        print("Please ensure all required features are present in the test data.")

    # After model training, add:
    try:
        # Create a dictionary with model metadata
        model_metadata = {
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'num_features': len(features),
            'num_classes': len(np.unique(y)),
            'feature_groups': feature_groups,
            'class_weights': class_weights
        }
        
        # Save metadata
        models_dir = 'models'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        metadata_path = os.path.join(models_dir, f'model_metadata_{timestamp}.joblib')
        dump(model_metadata, metadata_path)
        print(f"Model metadata saved to: {metadata_path}")
        
    except Exception as e:
        print(f"\nError saving model: {str(e)}")

def run_prediction():
    # Find the latest model
    print("\nFinding latest model...")
    models_dir = 'models'
    model_files = [f for f in os.listdir(models_dir) if f.startswith('random_forest_model_')]
    if not model_files:
        raise Exception("No model files found in models directory")
    
    latest_model_file = max(model_files)
    timestamp = latest_model_file.split('random_forest_model_')[1].split('.')[0]
    
    # Load the latest model and associated files
    print(f"\nLoading model and data from timestamp: {timestamp}")
    model = load(os.path.join(models_dir, latest_model_file))
    features = load(os.path.join(models_dir, f'features_{timestamp}.joblib'))
    
    # Find the latest scaler file instead of assuming timestamp
    scaler_files = [f for f in os.listdir(models_dir) if f.startswith('scaler_')]
    if not scaler_files:
        raise Exception("No scaler files found in models directory")
    latest_scaler_file = max(scaler_files)
    scaler = load(os.path.join(models_dir, latest_scaler_file))
    
    # Load and process test data
    print("\nLoading and processing test data...")
    test_df = pd.read_csv('test.csv')
    test_df = create_features(test_df)  # Make sure all necessary features are created
    
    # Ensure all features are present and in the correct order
    missing_features = set(features) - set(test_df.columns)
    if missing_features:
        print(f"Warning: Adding missing features: {missing_features}")
        for feature in missing_features:
            test_df[feature] = 0  # Initialize missing features with 0
    
    # Create X_test with features in the same order as during training
    X_test = test_df[features].copy()  # Use .copy() to avoid SettingWithCopyWarning
    
    # Scale the features
    X_test_scaled = scaler.transform(X_test)
    
    # Run prediction
    print("\nMaking predictions...")
    timestamp_now = datetime.now().strftime('%Y%m%d_%H%M%S')
    predictions_file = f'predictions_{timestamp_now}.csv'
    predictions = predict_and_save(model, X_test_scaled, test_df['uid'], predictions_file)
    print(f"\nPredictions saved to {predictions_file}")
    
    return predictions

if __name__ == "__main__":
    prepare_data()

# if __name__ == "__main__":
#     main()
