class GestureTrainer:
    # Add to __init__
    def __init__(self, input_size, num_classes, window_size=10, 
                 model_save_path="multi_gesture_model.npy",
                 loss_function='cross_entropy',
                 patience=10):  # Early stopping patience
        # ... existing code ...
        self.patience = patience
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    # New method for validation
    def validate(self, X_val, y_val):
        """
        Validate the model on validation data
        
        Parameters:
        -----------
        X_val : numpy.ndarray
            Validation features
        y_val : numpy.ndarray
            Validation labels
            
        Returns:
        --------
        tuple
            (val_loss, val_accuracy)
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call create_model() first.")
        
        # Make predictions
        y_pred = self.model.predict(X_val)
        
        # Calculate accuracy
        val_accuracy = np.mean(y_pred == y_val)
        
        # Calculate validation loss (this needs to be implemented in GestureClassificationNetwork)
        val_loss = self.model.calculate_loss(X_val, y_val)
        
        return val_loss, val_accuracy
    
    # Enhanced train method with validation
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=500, batch_size=128, learning_rate=0.001, 
              early_stopping=True):
        """
        Train the model with validation
        
        Parameters:
        -----------
        X_train : numpy.ndarray
            Training features
        y_train : numpy.ndarray
            Training labels
        X_val : numpy.ndarray
            Validation features (optional)
        y_val : numpy.ndarray
            Validation labels (optional)
        epochs : int
            Maximum number of training epochs
        batch_size : int
            Batch size for training
        learning_rate : float
            Learning rate for optimizer
        early_stopping : bool
            Whether to use early stopping based on validation loss
            
        Returns:
        --------
        dict
            Training history (losses, accuracies)
        """
        # Create model if not already created
        if self.model is None:
            self.create_model()
        
        # Convert to one-hot if needed (for cross-entropy)
        if self.loss_function == 'cross_entropy' and len(y_train.shape) == 1:
            y_train_one_hot = self.convert_to_one_hot(y_train)
        else:
            y_train_one_hot = y_train
            
        # Same for validation data if provided
        if X_val is not None and y_val is not None and self.loss_function == 'cross_entropy' and len(y_val.shape) == 1:
            y_val_one_hot = self.convert_to_one_hot(y_val)
        else:
            y_val_one_hot = y_val if y_val is not None else None
        
        # Initialize history tracking
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Reset early stopping counters
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Training loop
        print(f"Training model with {epochs} epochs, batch size {batch_size}, learning rate {learning_rate}")
        for epoch in range(epochs):
            # Train for one epoch
            train_loss = self.model.train_epoch(
                X_train=X_train,
                y_train=y_train_one_hot,
                batch_size=batch_size,
                learning_rate=learning_rate
            )
            
            # Calculate training accuracy
            train_pred = self.model.predict(X_train)
            train_acc = np.mean(train_pred == y_train)
            
            # Add to history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Validate if validation data provided
            if X_val is not None and y_val is not None:
                val_loss, val_acc = self.validate(X_val, y_val)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                
                # Early stopping check
                if early_stopping:
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.patience_counter = 0
                        # Save best model
                        self.save_best_model()
                    else:
                        self.patience_counter += 1
                        if self.patience_counter >= self.patience:
                            print(f"Early stopping at epoch {epoch+1}")
                            # Load best model
                            self.load_best_model()
                            break
            else:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        return history
    
    # Methods for saving/loading best model during training
    def save_best_model(self):
        # Save temporary best model
        best_model_path = os.path.splitext(self.model_save_path)[0] + "_best.npy"
        self.model.save_model(best_model_path)
    
    def load_best_model(self):
        # Load best model
        best_model_path = os.path.splitext(self.model_save_path)[0] + "_best.npy"
        if os.path.exists(best_model_path):
            self.model.load_model(best_model_path)
            # Delete temporary file
            os.remove(best_model_path)