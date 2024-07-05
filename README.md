
```markdown
# Human Activity Recognition Model

This repository contains a Long Short-Term Memory (LSTM) model for recognizing human activities using smartphone sensor data. The model is trained to classify activities into one of six categories: Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, and Laying.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Usage](#usage)
- [Testing the Model](#testing-the-model)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview

The Human Activity Recognition model is built using TensorFlow and Keras. It uses an LSTM architecture to detect and classify human activities based on data collected from smartphone sensors like accelerometers and gyroscopes.

## Dataset

The dataset used is the UCI HAR dataset, which contains recordings of 30 subjects performing activities of daily living (ADLs) while carrying a waist-mounted smartphone with embedded inertial sensors. The dataset can be downloaded from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones).

## Model Architecture

The model architecture consists of:
- Two LSTM layers with 100 units each.
- Dropout layers for regularization to prevent overfitting.
- A Dense layer with 6 units (one for each activity) and softmax activation for classification.

## Results

The model achieves a test accuracy of approximately 94% on the UCI HAR dataset. Detailed performance metrics can be found in the evaluation section of the notebook.

## Usage

To train and evaluate the model, follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/ziishanahmad/human-activity-recognition.git
   cd human-activity-recognition
   ```

2. Set up the environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download and unzip the UCI HAR dataset:
   ```bash
   !wget -q -O UCI_HAR_Dataset.zip https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip
   !unzip -q UCI_HAR_Dataset.zip -d ./UCI_HAR_Dataset
   ```

4. Run the Jupyter notebook to train and evaluate the model:
   ```bash
   jupyter notebook human-activity-recognition.ipynb
   ```

## Testing the Model

To test the model with some sample data, add the following code to the end of your Jupyter notebook:

```python
# Import necessary libraries
from tensorflow.keras.models import load_model
import numpy as np

# Load the trained model from the .h5 file
model = load_model('human_activity_recognition_model.h5')
# The model is now loaded and ready for making predictions

# Select a few samples from the test set for demonstration
num_samples = 5  # Number of samples to test
sample_indices = np.random.choice(X_test_reshaped.shape[0], num_samples, replace=False)
# Randomly select `num_samples` indices from the test set

# Extract the selected samples and their true labels
sample_data = X_test_reshaped[sample_indices]
# Extract the selected samples from the test data
true_labels = y_test[sample_indices]
# Extract the true labels for the selected samples

# Use the loaded model to make predictions on the sample data
predictions = model.predict(sample_data)
# The model outputs the class probabilities for each sample
predicted_classes = np.argmax(predictions, axis=1)
# Convert the class probabilities to class labels by taking the argmax

# Define the mapping from class indices to activity labels
activity_labels = ['Walking', 'Walking Upstairs', 'Walking Downstairs', 'Sitting', 'Standing', 'Laying']
# This is the same order as used in the dataset

# Print the results
print("Sample Data Predictions:")
for i in range(num_samples):
    print(f"Sample {i+1}:")
    print(f"  True Label: {activity_labels[true_labels[i][0]]}")
    print(f"  Predicted Label: {activity_labels[predicted_classes[i]]}")
    print()
```

## Requirements

- Python 3.7+
- TensorFlow 2.x
- Keras
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

Install the requirements using:
```bash
pip install -r requirements.txt
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For any questions or feedback, please contact:

- **Name:** Zeeshan Ahmad
- **Email:** ziishanahmad@gmail.com
- **GitHub:** [ziishanahmad](https://github.com/ziishanahmad)
- **LinkedIn:** [ziishanahmad](https://www.linkedin.com/in/ziishanahmad/)

---
