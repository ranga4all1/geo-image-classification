# Evaluate and Predict using geo classification model

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.applications.xception import preprocess_input
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

class GeoClassifier:
    def __init__(self, model_path: str, input_size: int = 299):
        """
        Initialize the classifier with a trained model.
        
        Args:
            model_path: Path to the saved Keras model
            input_size: Input size for the model (default: 299 for Xception)
        """
        self.input_size = input_size
        self.classes = [
            'buildings', 'forest', 'glacier',
            'mountain', 'sea', 'street'
        ]
        
        try:
            self.model = keras.models.load_model(model_path)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            raise Exception(f"Failed to load model from {model_path}: {str(e)}")
    
    def evaluate_test_set(self, test_dir: str, batch_size: int = 32) -> Tuple[float, float]:
        """
        Evaluate the model on a test dataset.
        
        Args:
            test_dir: Directory containing test images organized in class folders
            batch_size: Batch size for evaluation
            
        Returns:
            Tuple of (loss, accuracy)
        """
        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"Test directory not found: {test_dir}")
            
        test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
        
        test_ds = test_gen.flow_from_directory(
            test_dir,
            target_size=(self.input_size, self.input_size),
            batch_size=batch_size,
            shuffle=False
        )
        
        print("\nEvaluating model on test dataset...")
        loss, accuracy = self.model.evaluate(test_ds)
        print(f"\nTest Results:")
        print(f"Loss: {loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        
        return loss, accuracy
    
    def predict_image(self, image_path: str, display: bool = True) -> Dict[str, float]:
        """
        Predict the class of a single image.
        
        Args:
            image_path: Path to the image file
            display: Whether to display the image with predictions
            
        Returns:
            Dictionary of class probabilities
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        # Load and preprocess image
        try:
            img = load_img(image_path, target_size=(self.input_size, self.input_size))
            x = np.array(img)
            X = np.array([x])
            X = preprocess_input(X)
        except Exception as e:
            raise Exception(f"Error processing image {image_path}: {str(e)}")
            
        # Make prediction
        pred = self.model.predict(X, verbose=0)
        
        # Convert to probabilities
        probabilities = tf.nn.softmax(pred[0]).numpy()
        
        # Create results dictionary
        results = dict(zip(self.classes, probabilities))
        
        # Sort by probability
        results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
        
        if display:
            self._display_prediction(img, results)
            
        return results
    
    def predict_batch(self, image_dir: str, batch_size: int = 32) -> List[Dict[str, float]]:
        """
        Predict classes for all images in a directory.
        
        Args:
            image_dir: Directory containing images
            batch_size: Batch size for predictions
            
        Returns:
            List of dictionaries containing class probabilities for each image
        """
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Directory not found: {image_dir}")
            
        # Create data generator for prediction
        pred_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
        
        pred_ds = pred_gen.flow_from_directory(
            image_dir,
            target_size=(self.input_size, self.input_size),
            batch_size=batch_size,
            shuffle=False
        )
        
        # Make predictions
        predictions = self.model.predict(pred_ds)
        
        # Convert to probabilities
        probabilities = tf.nn.softmax(predictions).numpy()
        
        # Create list of results
        results = []
        for pred in probabilities:
            results.append(dict(zip(self.classes, pred)))
            
        return results
    
    def _display_prediction(self, img, results: Dict[str, float]):
        """Display the image and prediction results."""
        plt.figure(figsize=(10, 5))
        
        # Display image
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title('Input Image')
        
        # Display predictions
        plt.subplot(1, 2, 2)
        classes = list(results.keys())
        probs = list(results.values())
        
        y_pos = np.arange(len(classes))
        plt.barh(y_pos, probs)
        plt.yticks(y_pos, classes)
        plt.xlabel('Probability')
        plt.title('Predictions')
        
        plt.tight_layout()
        plt.show()

def main():
    """Main function to demonstrate usage."""
    # Initialize classifier
    try:
        classifier = GeoClassifier('geo-model.keras')
        
        # Evaluate on test set
        test_loss, test_acc = classifier.evaluate_test_set(
            '../data/seg_test/seg_test'
        )
        
        # Example single image prediction
        image_path = '../data/seg_pred/seg_pred/10054.jpg'
        results = classifier.predict_image(image_path)
        
        print("\nPrediction Results:")
        for class_name, prob in results.items():
            print(f"{class_name}: {prob:.4f}")
        
        # Example batch prediction
        batch_results = classifier.predict_batch('../data/seg_pred')
        print(f"\nProcessed {len(batch_results)} images in batch mode")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()


"""
Usage:

# Simple single image prediction
classifier = GeoClassifier('geo-model.keras')
results = classifier.predict_image('path/to/image.jpg')

# Batch processing
results = classifier.predict_batch('path/to/image/directory')

# Evaluation
loss, acc = classifier.evaluate_test_set('path/to/test/directory')

"""
