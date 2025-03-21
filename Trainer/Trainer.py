# G:\AnacondaEnvironment\directml\MLProject\ml-project\Trainer\Trainer.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from Dataset import augment_dataset, stratified_train_test_split
from Network.GestureClassificationNetwork import GestureClassificationNetwork

class GestureTrainer:
    """
    Handles the training, validation, and evaluation of gesture recognition models.
    """
    
    def __init__(self, input_size, num_classes, window_size=10, 
                 model_save_path="multi_gesture_model.npy",
                 loss_function='cross_entropy'):
        """
        Initialize the trainer.
        
        Parameters:
        -----------
        input_size : int
            Size of input features
        num_classes : int
            Number of gesture classes
        window_size : int
            Size of sliding window for temporal data
        model_save_path : str
            Path to save the model
        loss_function : str
            Type of loss function ('cross_entropy' or 'mse')
        """
        self.input_size = input_size
        self.num_classes = num_classes
        self.window_size = window_size
        self.model_save_path = model_save_path
        self.loss_function = loss_function
        
        # Initialize model
        self.model = GestureClassificationNetwork(
            input_size=input_size,
            num_gestures=num_classes,
            window_size=window_size,
            loss_function=loss_function
        )
        
        # Paths for saving model related files
        self.mapping_save_path = os.path.splitext(model_save_path)[0] + "_mapping.npy"
        self.params_save_path = os.path.splitext(model_save_path)[0] + "_params.npy"
        
        # Metrics tracking
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_accuracy = 0
        
        # Label mapping
        self.label_mapping = None
    
    def prepare_data(self, dataset, train_landmarks, train_labels, val_landmarks=None, val_labels=None):
        """
        Prepare data for training and validation.
        
        Parameters:
        -----------
        dataset : GestureDataset
            The dataset class to use for data preparation
        train_landmarks : list
            Landmark data for training
        train_labels : list
            Labels for training
        val_landmarks : list, optional
            Landmark data for validation (if None, split from training)
        val_labels : list, optional
            Labels for validation (if None, split from training)
            
        Returns:
        --------
        tuple
            (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        print("Preparing training data...")
        
        # Get label mapping first (so validation uses the same mapping)
        X_train_temp, y_train_temp = dataset.prepare_windows(train_landmarks, train_labels)
        self.label_mapping = dataset.get_label_mapping()
        
        # Apply data augmentation to training data
        train_landmarks, train_labels = augment_dataset(train_landmarks, train_labels)
        
        # Create windows for training data
        X_train, y_train = dataset.prepare_windows(train_landmarks, train_labels)
        
        # Create windows for validation data if provided
        if val_landmarks is not None and val_labels is not None:
            X_val, y_val = dataset.prepare_windows(val_landmarks, val_labels)
            # Split validation into validation and test sets
            X_val, X_test, y_val, y_test = stratified_train_test_split(
                X_val, y_val, test_size=0.3, random_seed=42
            )
        else:
            # Split training data for validation and test
            X_train, X_val, y_train, y_val = stratified_train_test_split(
                X_train, y_train, test_size=0.2, random_seed=42
            )
            # Split validation into validation and test sets
            X_val, X_test, y_val, y_test = stratified_train_test_split(
                X_val, y_val, test_size=0.5, random_seed=42
            )
        
        # Convert to one-hot for training (the train method will handle this)
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def train_with_validation(self, X_train, y_train, X_val, y_val, epochs=500, batch_size=16, learning_rate=0.0001):
        """
        Train the model with validation.
        
        Parameters:
        -----------
        X_train : numpy.ndarray
            Training features
        y_train : numpy.ndarray
            Training labels
        X_val : numpy.ndarray
            Validation features
        y_val : numpy.ndarray
            Validation labels
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        learning_rate : float
            Learning rate for optimizer
            
        Returns:
        --------
        self
        """
        print(f"Training model with {epochs} epochs, batch size {batch_size}, learning rate {learning_rate}")
        
        # Reset metrics
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_accuracy = 0
        best_model_state = None
        
        # Convert labels to one-hot for training if needed
        if len(y_train.shape) == 1:
            y_train_one_hot = np.zeros((y_train.shape[0], self.num_classes))
            y_train_one_hot[np.arange(y_train.shape[0]), y_train] = 1
        else:
            y_train_one_hot = y_train
            
        # Training loop with validation after each epoch
        for epoch in range(epochs):
            # Train one epoch using your existing train method
            # We'll modify it slightly to train just one epoch
            self.model.train(
                X_train=X_train, 
                y_train=y_train_one_hot,
                epochs=1,  # Just one epoch at a time
                batch_size=batch_size,
                learning_rate=learning_rate
            )
            
            # Evaluate on training set
            train_pred = self.model.predict(X_train)
            if len(y_train.shape) > 1:  # If one-hot encoded
                train_true = np.argmax(y_train, axis=1)
            else:
                train_true = y_train
            train_acc = np.mean(train_pred == train_true)
            
            # Evaluate on validation set
            val_pred = self.model.predict(X_val)
            val_acc = np.mean(val_pred == y_val)
            
            # Store metrics
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save model if validation accuracy improves
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                # Save model state
                self.model.save_model(self.model_save_path + ".best")
                print(f"  Validation accuracy improved to {val_acc:.4f} - Saving model")
                
        # Load best model
        if os.path.exists(self.model_save_path + ".best"):
            self.model.load_model(self.model_save_path + ".best")
            print(f"Loaded best model with validation accuracy: {self.best_val_accuracy:.4f}")
            # Remove temporary file
            os.remove(self.model_save_path + ".best")
        
        return self
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Parameters:
        -----------
        X_test : numpy.ndarray
            Test features
        y_test : numpy.ndarray
            Test labels
            
        Returns:
        --------
        dict
            Dictionary with test metrics
        """
        print("Evaluating model on test set...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate accuracy
        accuracy = np.mean(y_pred == y_test)
        print(f"Test accuracy: {accuracy:.4f}")
        
        # Calculate per-class metrics
        class_metrics = {}
        for i in range(self.num_classes):
            # Get indices for this class
            true_indices = np.where(y_test == i)[0]
            if len(true_indices) > 0:
                # Class accuracy
                class_acc = np.mean(y_pred[true_indices] == y_test[true_indices])
                
                # Get confusion matrix elements
                tp = np.sum((y_pred[true_indices] == i))
                fp = np.sum((y_pred == i) & (y_test != i))
                fn = np.sum((y_pred != i) & (y_test == i))
                
                # Calculate precision, recall, F1
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                # Get class name
                class_name = self.get_index_to_label().get(i, f"Class {i}")
                
                # Store metrics
                class_metrics[class_name] = {
                    'accuracy': class_acc,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'support': len(true_indices)
                }
                
                print(f"  {class_name}: Acc={class_acc:.4f}, F1={f1:.4f}, Support={len(true_indices)}")
        
        # Calculate confusion matrix
        cm = np.zeros((self.num_classes, self.num_classes), dtype=int)
        for i in range(len(y_test)):
            cm[y_test[i], y_pred[i]] += 1
        
        return {
            'accuracy': accuracy,
            'class_metrics': class_metrics,
            'confusion_matrix': cm
        }
    
    def save_model(self, included_landmarks):
        """
        Save the model and related data.
        
        Parameters:
        -----------
        included_landmarks : list
            List of included landmarks indices
            
        Returns:
        --------
        self
        """
        # Save model
        self.model.save_model(self.model_save_path)
        print(f"Model saved to {self.model_save_path}")
        
        # Save label mapping
        np.save(self.mapping_save_path, self.label_mapping)
        print(f"Gesture mapping saved to {self.mapping_save_path}")
        
        # Save feature parameters
        np.save(self.params_save_path, {
            'window_size': self.window_size,
            'feature_size': self.input_size,
            'included_landmarks': included_landmarks
        })
        print(f"Feature parameters saved to {self.params_save_path}")
        
        return self
    
    def load_model(self):
        """
        Load the model and related data.
        
        Returns:
        --------
        self
        """
        self.model.load_model(self.model_save_path)
        print(f"Model loaded from {self.model_save_path}")
        
        if os.path.exists(self.mapping_save_path):
            self.label_mapping = np.load(self.mapping_save_path, allow_pickle=True).item()
            print(f"Gesture mapping loaded from {self.mapping_save_path}")
        
        return self
    
    def get_label_to_index(self):
        """Get mapping from label to index."""
        return self.label_mapping
    
    def get_index_to_label(self):
        """Get mapping from index to label."""
        return {idx: label for label, idx in self.label_mapping.items()}
    
    def plot_training_history(self, save_path="training_history.png"):
        """
        Plot training history.
        
        Parameters:
        -----------
        save_path : str
            Path to save the plot
            
        Returns:
        --------
        self
        """
        if not self.train_accuracies:
            print("No training history to plot")
            return self
        
        plt.figure(figsize=(10, 6))
        
        # Plot accuracy
        plt.plot(self.train_accuracies, label='Training Accuracy')
        plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
        
        return self
    
    def plot_confusion_matrix(self, confusion_matrix, save_path="confusion_matrix.png"):
        """
        Plot confusion matrix.
        
        Parameters:
        -----------
        confusion_matrix : numpy.ndarray
            Confusion matrix
        save_path : str
            Path to save the plot
            
        Returns:
        --------
        self
        """
        # Create labels for the plot
        class_names = [self.get_index_to_label().get(i, f"Class {i}") for i in range(self.num_classes)]
        
        # Plot the confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
        
        return self