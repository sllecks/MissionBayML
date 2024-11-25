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
import os
from datetime import datetime

# Load the data
def load_data():
    train_df = pd.read_csv('train.csv')
    return train_df

def create_features(df):
    # Basic pitch movement features
    df['total_break'] = np.sqrt(df['pfx_x']**2 + df['pfx_z']**2)
    df['break_angle'] = np.arctan2(df['pfx_z'], df['pfx_x'])
    
    # Game situation features
    df['opposite_hand'] = df['is_lhp'] != df['is_lhb']
    df['on_base'] = df['on_1b'] + df['on_2b'] + df['on_3b']
    df['late_game'] = df['inning'] >= 7
    df['pressure'] = (df['outs_when_up'] == 2) & ((df['on_2b'] == 1) | (df['on_3b'] == 1))
    df['clutch'] = df['pressure'] & (df['late_game'] == 1)
    
    # Pitcher state features
    df['fatigue'] = df['pitch_number'] / 110
    
    # Advanced physics features
    df['momentum_x'] = df['release_speed'] * df['release_pos_x']
    df['momentum_z'] = df['release_speed'] * df['release_pos_z']
    df['total_acceleration'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
    df['pitch_break_horizontal'] = df['vx0'] - df['pfx_x']
    df['pitch_break_vertical'] = df['vz0'] - df['pfx_z']
    
    # Rolling statistics
    df = df.sort_values(by=['uid', 'pitch_number'])
    df['avg_speed_last_5'] = df.groupby('uid')['release_speed'].rolling(window=5, min_periods=1).mean().reset_index(drop=True)
    df['avg_speed_last_10'] = df.groupby('uid')['release_speed'].rolling(window=10, min_periods=1).mean().reset_index(drop=True)
    df['avg_spin_last_5'] = df.groupby('uid')['release_spin_rate'].rolling(window=5, min_periods=1).mean().reset_index(drop=True)
    df['avg_spin_last_10'] = df.groupby('uid')['release_spin_rate'].rolling(window=10, min_periods=1).mean().reset_index(drop=True)
    
    # Add interaction features
    df['speed_spin_ratio'] = df['release_speed'] / df['release_spin_rate']
    df['vertical_approach_angle'] = np.arctan2(df['vz0'], df['vx0'])
    
    # Add count features
    df['favorable_count'] = ((df['balls'] > df['strikes']) | 
                           ((df['balls'] == 3) & (df['strikes'] < 3)))
    
    # Add more advanced rolling stats
    df['spin_efficiency'] = df['release_spin_rate'] / df.groupby('uid')['release_spin_rate'].transform('max')
    df['pitch_tunneling'] = df.groupby('uid')['release_pos_x'].diff().abs() + df.groupby('uid')['release_pos_z'].diff().abs()
    
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
    hit_outcomes = ['single', 'double', 'triple', 'home_run']
    df['hit_type'] = df['outcome']
    df.loc[~df['outcome'].isin(hit_outcomes), 'hit_type'] = 'other'
    
    # Encode categorical variables
    categorical_cols = ['is_lhp', 'is_lhb', 'pitch_type', 'hit_type']
    le_dict = {}
    for col in categorical_cols:
        le_dict[col] = LabelEncoder()
        df[col] = le_dict[col].fit_transform(df[col].astype(str))
    
    # Combine all features
    features = []
    for group in feature_groups.values():
        features.extend(group)
    features = list(set(features))  # Remove any duplicates
    
    # Handle missing values
    df[features] = df[features].fillna(df[features].mean())

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    y = df['hit_type']

    # SMOTE for handling class imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Feature selection using LASSO
    selector = SelectFromModel(Lasso(alpha=0.01))
    X_selected = selector.fit_transform(X_resampled, y_resampled)
    selected_features = [features[i] for i in range(len(features)) 
                        if selector.get_support()[i]]

    return X_selected, y_resampled, selected_features, le_dict['hit_type'], scaler, feature_groups

def train_model(X_train, y_train, X_test, y_test, features, feature_groups=None):
    from sklearn.ensemble import RandomForestClassifier
    from collections import Counter
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, roc_curve, auc
    from sklearn.model_selection import cross_val_score
    
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

    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
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
    
    return model

def predict_and_save(model, test_df, features, scaler, output_file):
    # Create features first
    test_df = create_features(test_df)

    # Check for missing features and handle them
    missing_features = [f for f in features if f not in test_df.columns]
    if missing_features:
        print(f"Warning: Missing features in test data: {missing_features}")
        # You can choose to:
        # 1. Raise an error: raise ValueError(f"Missing required features: {missing_features}")
        # 2. Impute missing values: test_df[missing_features] = test_df[missing_features].fillna(test_df[missing_features].mean())
        # 3. Exclude samples with missing features: test_df = test_df.dropna(subset=missing_features)

    # Encode categorical columns using the same LabelEncoder as in training
    categorical_cols = ['is_lhp', 'is_lhb', 'pitch_type']
    for col in categorical_cols:
        if col in features:
            le = le_dict[col]  # Use the same LabelEncoder
            test_df[col] = le.transform(test_df[col].astype(str))

    # Handle missing values (if not handled above)
    test_df[features] = test_df[features].fillna(test_df[features].mean())

    # Scale features
    X_test = scaler.transform(test_df[features])
    
    # Get probability predictions
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
        'uid': test_df['uid'],
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
    # Load and prepare training data
    print("Loading and preparing training data...")
    df = load_data()
    df = create_features(df)
    X, y, features, label_encoder, scaler, feature_groups = prepare_data(df)
    
    # Split the data
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train the model
    print("\nTraining model...")
    model = train_model(X_train, y_train, X_test, y_test, features, feature_groups)
    
    # Load and process test data
    print("\nLoading test data...")
    test_df = pd.read_csv('test.csv')
    
    try:
        # Make predictions and save to CSV
        print("\nMaking predictions...")
        predictions = predict_and_save(model, test_df, features, scaler, 'predictions.csv')
        
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

if __name__ == "__main__":
    main()
