"""
Iris Flower Classification for VS Code
This script was adapted from Jupyter Lab to work in VS Code
"""

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import joblib
import warnings
warnings.filterwarnings('ignore')


plt.style.use('ggplot')  
try:
    sns.set_palette("husl")
except Exception:
    pass  

# Initialize global variables
scaler = None
models = {}
voting_clf = None

def main():

    print("=== LOADING AND EXPLORING DATASET ===")
    iris = load_iris()
    X = iris.data
    y = iris.target

    df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                    columns=iris['feature_names'] + ['target'])

    print("Dataset shape:", df.shape)
    print("\nColumn information:")
    print(df.info())
    print("\nSummary statistics:")
    print(df.describe())

    print("\nSample data:")
    print(df.head())

    print("\n=== DATA VISUALIZATION ===")
    visualize_data(df, iris)

    
    print("\n=== PREPARING DATA FOR TRAINING ===")
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = prepare_data(X, y)
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")

    print("\n=== TRAINING CLASSIFICATION MODELS ===")
    global models
    models, results, predictions = train_models(X_train, X_test, y_train, y_test, 
                                             X_train_scaled, X_test_scaled, iris)

    print("\n=== MODEL PERFORMANCE VISUALIZATION ===")
    visualize_model_performance(models, results, predictions, y_test, iris)

    print("\n=== FEATURE IMPORTANCE ===")
    feature_importance_analysis(models, iris)

    print("\n=== HYPERPARAMETER TUNING (KNN) ===")
    best_knn = tune_knn(X_train_scaled, y_train, X_test_scaled, y_test)

    print("\n=== DECISION BOUNDARIES VISUALIZATION ===")
    visualize_decision_boundaries(X, y, models)

    print("\n=== CUSTOM PREDICTION ===")
    custom_prediction_function(models, iris, df, scaler)

    print("\n=== ENSEMBLE VOTING CLASSIFIER ===")
    global voting_clf
    voting_accuracy, results = ensemble_voting(X_train_scaled, y_train, X_test_scaled, 
                                            y_test, results, iris)

    print("\n=== SAVING BEST MODEL ===")
    save_best_model(models, results, voting_clf, scaler)

    print("\n=== LOADING AND USING SAVED MODEL ===")
    load_and_use_model(iris)

def visualize_data(df, iris):
    """Visualize the iris dataset"""
    
    # Ask user which visualizations they want to see
    print("\nSelect which visualizations to show:")
    print("1. Pairplot")
    print("2. Boxplots")
    print("3. Correlation heatmap")
    print("4. Feature distributions")
    print("5. All visualizations")
    print("6. None (skip visualizations)")
    
    choice = input("Enter your choice (1-6): ")
    
    # Pairplot to see relationships between features
    if choice in ['1', '5']:
        plt.figure(figsize=(12, 10))
        sns.pairplot(df, hue='target', palette='husl')
        plt.suptitle('Iris Dataset - Feature Relationships', y=1.02)
        plt.show()

    # Boxplots for each feature
    if choice in ['2', '5']:
        plt.figure(figsize=(15, 10))
        for i, column in enumerate(iris.feature_names):
            plt.subplot(2, 2, i+1)
            sns.boxplot(x='target', y=column, data=df, palette='husl')
            plt.xticks([0, 1, 2], iris.target_names)
            plt.title(f'{column} by Species')
        plt.tight_layout()
        plt.show()

    # Correlation heatmap
    if choice in ['3', '5']:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Heatmap')
        plt.show()

    # Feature Distribution
    if choice in ['4', '5']:
        plt.figure(figsize=(15, 10))
        for i, column in enumerate(iris.feature_names):
            plt.subplot(2, 2, i+1)
            for species in range(3):
                sns.kdeplot(df[df['target'] == species][column], label=iris.target_names[species], shade=True)
            plt.title(f'{column} Distribution by Species')
            plt.legend()
        plt.tight_layout()
        plt.show()

def prepare_data(X, y):
    """Prepare data for model training"""
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scale the features
    global scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled

def train_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, iris):
    """Train multiple classification models"""
    # Initialize models
    models = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'SVM': SVC(kernel='rbf', random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    # Train and evaluate each model
    results = {}
    predictions = {}

    for model_name, model in models.items():
        # Use scaled data for KNN and SVM
        if model_name in ['KNN', 'SVM']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Store results
        accuracy = accuracy_score(y_test, y_pred)
        results[model_name] = accuracy
        predictions[model_name] = y_pred
        
        # Print results
        print(f"\n{model_name} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    return models, results, predictions

def visualize_model_performance(models, results, predictions, y_test, iris):
    """Visualize model performance"""
    # Ask user which visualizations they want to see
    print("\nSelect which model performance visualizations to show:")
    print("1. Model accuracy comparison")
    print("2. Confusion matrices")
    print("3. Both")
    print("4. None (skip visualizations)")
    
    choice = input("Enter your choice (1-4): ")

    # Compare model accuracies
    if choice in ['1', '3']:
        plt.figure(figsize=(10, 6))
        models_names = list(results.keys())
        accuracies = list(results.values())
        bars = plt.bar(models_names, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        plt.ylim(0.8, 1.01)
        plt.title('Model Comparison - Accuracy')
        plt.ylabel('Accuracy')
        for i, v in enumerate(accuracies):
            plt.text(i, v + 0.005, f'{v:.3f}', ha='center')
        plt.show()

    # Confusion matrices
    if choice in ['2', '3']:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()

        for idx, (model_name, predictions) in enumerate(predictions.items()):
            cm = confusion_matrix(y_test, predictions)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                        xticklabels=iris.target_names,
                        yticklabels=iris.target_names)
            axes[idx].set_title(f'{model_name} Confusion Matrix')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        plt.tight_layout()
        plt.show()

def feature_importance_analysis(models, iris):
    """Analyze feature importance"""
    # Ask user which feature importance visualizations they want to see
    print("\nSelect which feature importance visualizations to show:")
    print("1. Decision Tree feature importance")
    print("2. Random Forest feature importance")
    print("3. Both")
    print("4. None (skip visualizations)")
    
    choice = input("Enter your choice (1-4): ")

    # Decision Tree feature importance
    if choice in ['1', '3']:
        plt.figure(figsize=(10, 6))
        dt_importance = models['Decision Tree'].feature_importances_
        feature_importance = pd.DataFrame({'feature': iris.feature_names, 'importance': dt_importance})
        feature_importance = feature_importance.sort_values('importance', ascending=False)

        sns.barplot(x='importance', y='feature', data=feature_importance, palette='viridis')
        plt.title('Feature Importance (Decision Tree)')
        plt.show()

    # Random Forest feature importance
    if choice in ['2', '3']:
        plt.figure(figsize=(10, 6))
        rf_importance = models['Random Forest'].feature_importances_
        feature_importance_rf = pd.DataFrame({'feature': iris.feature_names, 'importance': rf_importance})
        feature_importance_rf = feature_importance_rf.sort_values('importance', ascending=False)

        sns.barplot(x='importance', y='feature', data=feature_importance_rf, palette='viridis')
        plt.title('Feature Importance (Random Forest)')
        plt.show()

def tune_knn(X_train_scaled, y_train, X_test_scaled, y_test):
    """Hyperparameter tuning for KNN"""
    # Define parameter grid for KNN
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    # Perform grid search
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    grid_search.fit(X_train_scaled, y_train)

    # Print best parameters
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)

    # Use the best model
    best_knn = grid_search.best_estimator_
    best_knn_pred = best_knn.predict(X_test_scaled)
    print("Best KNN Accuracy:", accuracy_score(y_test, best_knn_pred))

    # Ask user if they want to see the hyperparameter tuning results
    show_results = input("\nShow hyperparameter tuning results? (y/n): ")
    if show_results.lower() == 'y':
        # Visualize hyperparameter tuning results
        results_df = pd.DataFrame(grid_search.cv_results_)
        # Create pivot table if there are enough results
        if len(grid_search.cv_results_['param_weights']) > 1:
            pivot_table = results_df.pivot_table(
                values='mean_test_score',
                index='param_n_neighbors',
                columns='param_weights'
            )

            plt.figure(figsize=(10, 6))
            sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt='.3f')
            plt.title('KNN Hyperparameter Tuning - Mean Test Score')
            plt.show()
        else:
            # Simple line plot
            plt.figure(figsize=(10, 6))
            plt.plot(results_df['param_n_neighbors'], results_df['mean_test_score'], '-o')
            plt.title('KNN Hyperparameter Tuning - Mean Test Score')
            plt.xlabel('n_neighbors')
            plt.ylabel('Mean Test Score')
            plt.grid(True)
            plt.show()
            
    return best_knn

def visualize_decision_boundaries(X, y, models):
    """Visualize decision boundaries using PCA"""
    # Ask user if they want to see the decision boundaries
    show_boundaries = input("\nShow decision boundaries? (y/n): ")
    if show_boundaries.lower() != 'y':
        return
        
    # Make sure scaler is defined
    global scaler
    if scaler is None:
        # Create a new scaler if one doesn't exist
        scaler = StandardScaler()
        scaler.fit(X)
        
    # Reduce dimensions using PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Create meshgrid for decision boundary
    h = 0.02  # step size in the mesh
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Plot decision boundaries for each model
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()

    for idx, (model_name, model) in enumerate(models.items()):
        # Train model on PCA-transformed data
        if model_name in ['KNN', 'SVM']:
            model.fit(scaler.fit_transform(X_pca), y)
            Z = model.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
        else:
            model.fit(X_pca, y)
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        axes[idx].contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.Spectral)
        axes[idx].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.Spectral, edgecolor='black')
        axes[idx].set_title(f'{model_name} Decision Boundary')
        axes[idx].set_xlabel('PCA Component 1')
        axes[idx].set_ylabel('PCA Component 2')

    plt.tight_layout()
    plt.show()

def custom_prediction_function(models, iris, df, scaler):
    """Custom prediction function (replaces interactive widget)"""
    # Get user input for flower measurements
    print("\nEnter flower measurements for prediction:")
    try:
        sepal_length = float(input("Sepal Length (cm) [4.0-8.0]: "))
        sepal_width = float(input("Sepal Width (cm) [2.0-4.5]: "))
        petal_length = float(input("Petal Length (cm) [1.0-7.0]: "))
        petal_width = float(input("Petal Width (cm) [0.1-2.5]: "))
    except ValueError:
        print("Invalid input. Using default values.")
        sepal_length, sepal_width, petal_length, petal_width = 5.8, 3.0, 4.0, 1.2
    
    # Create new data point
    new_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Scale the data
    new_data_scaled = scaler.transform(new_data)
    
    # Make predictions with each model
    predictions = {}
    for model_name, model in models.items():
        if model_name in ['KNN', 'SVM']:
            pred = model.predict(new_data_scaled)
        else:
            pred = model.predict(new_data)
        predictions[model_name] = iris.target_names[pred[0]]
    
    # Display predictions
    print("\nPredictions for the given measurements:")
    for model_name, prediction in predictions.items():
        print(f"{model_name}: {prediction}")
    
    # Plot the new point on a scatter plot
    show_plot = input("\nShow prediction visualization? (y/n): ")
    if show_plot.lower() == 'y':
        plt.figure(figsize=(10, 6))
        plt.scatter(df['sepal length (cm)'], df['petal length (cm)'], 
                    c=df['target'], cmap='viridis', alpha=0.7)
        plt.scatter(sepal_length, petal_length, c='red', s=200, marker='*', 
                    edgecolor='black', linewidth=2)
        plt.xlabel('Sepal Length (cm)')
        plt.ylabel('Petal Length (cm)')
        plt.title('Your Flower Measurement (Red Star)')
        plt.colorbar(label='Species')
        plt.show()

def ensemble_voting(X_train_scaled, y_train, X_test_scaled, y_test, results, iris):
    """Create and evaluate ensemble voting classifier"""
    # Create voting classifier
    global voting_clf
    voting_clf = VotingClassifier(
        estimators=[
            ('knn', KNeighborsClassifier(n_neighbors=5)),
            ('svm', SVC(probability=True, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
        ],
        voting='soft'
    )

    # Fit and predict
    voting_clf.fit(X_train_scaled, y_train)
    voting_pred = voting_clf.predict(X_test_scaled)

    # Evaluate
    voting_accuracy = accuracy_score(y_test, voting_pred)
    print(f"Voting Classifier Accuracy: {voting_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, voting_pred, target_names=iris.target_names))

    # Add to results comparison
    results['Voting Classifier'] = voting_accuracy

    # Ask user if they want to see the updated comparison plot
    show_plot = input("\nShow updated model comparison? (y/n): ")
    if show_plot.lower() == 'y':
        # Update comparison plot
        plt.figure(figsize=(12, 6))
        models_names = list(results.keys())
        accuracies = list(results.values())
        bars = plt.bar(models_names, accuracies)
        plt.ylim(0.8, 1.01)
        plt.title('Model Comparison - Including Voting Classifier')
        plt.ylabel('Accuracy')
        for i, v in enumerate(accuracies):
            plt.text(i, v + 0.005, f'{v:.3f}', ha='center')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    return voting_accuracy, results

def save_best_model(models, results, voting_clf, scaler):
    """Save the best performing model"""
    # Save the best performing model
    best_model_name = max(results, key=results.get)
    
    # Check if voting_clf exists
    if best_model_name == 'Voting Classifier' and voting_clf is None:
        print("Error: Voting Classifier selected but not defined.")
        best_model_name = max({k: v for k, v in results.items() if k != 'Voting Classifier'}, 
                           key=results.get)
        print(f"Using {best_model_name} instead.")
    
    best_model = models[best_model_name] if best_model_name != 'Voting Classifier' else voting_clf
    
    # Get directory from user
    save_dir = input("\nEnter directory to save model (leave blank for current directory): ")
    if not save_dir:
        save_dir = "."
    
    model_path = f"{save_dir}/best_iris_model.joblib"
    scaler_path = f"{save_dir}/iris_scaler.joblib"
    
    joblib.dump(best_model, model_path)
    print(f"Best model ({best_model_name}) saved as '{model_path}'")

    # Save the scaler
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved as '{scaler_path}'")

def load_and_use_model(iris):
    """Load and use the saved model"""
    try:
        # Ask user for model path
        model_path = input("\nEnter path to saved model (leave blank for 'best_iris_model.joblib'): ")
        if not model_path:
            model_path = "best_iris_model.joblib"
            
        scaler_path = input("Enter path to saved scaler (leave blank for 'iris_scaler.joblib'): ")
        if not scaler_path:
            scaler_path = "iris_scaler.joblib"
        
        # Load the saved model and scaler
        loaded_model = joblib.load(model_path)
        loaded_scaler = joblib.load(scaler_path)

        # Ask user for flower measurements
        print("\nEnter flower measurements for prediction using loaded model:")
        try:
            sepal_length = float(input("Sepal Length (cm) [4.0-8.0]: "))
            sepal_width = float(input("Sepal Width (cm) [2.0-4.5]: "))
            petal_length = float(input("Petal Length (cm) [1.0-7.0]: "))
            petal_width = float(input("Petal Width (cm) [0.1-2.5]: "))
        except ValueError:
            print("Invalid input. Using default values.")
            sepal_length, sepal_width, petal_length, petal_width = 5.9, 3.0, 5.1, 1.8
            
        # Make a prediction with the loaded model
        test_flower = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        test_flower_scaled = loaded_scaler.transform(test_flower)
        prediction = loaded_model.predict(test_flower_scaled)
        print(f"Predicted species: {iris.target_names[prediction[0]]}")
    except FileNotFoundError:
        print("Error: Model or scaler file not found.")
    except Exception as e:
        print(f"Error loading or using model: {e}")

if __name__ == "__main__":
    main()