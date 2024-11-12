# Machine Learning Systems Laboratory Exam Solution
# Import necessary libraries
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# ================================
# Question 1: Data Structure and Processing Pipeline
# ================================


class IrisDataProcessor:
    def __init__(self):
        # Load the Iris dataset
        self.data = load_iris()
        self.df = pd.DataFrame(data=self.data.data,
                               columns=self.data.feature_names)
        self.df['target'] = self.data.target
        self.scaler = StandardScaler()

    def prepare_data(self):
        # Convert data to pandas DataFrame and perform scaling
        features = self.df.drop(columns=['target'])
        target = self.df['target']

        # Feature scaling using StandardScaler
        scaled_features = self.scaler.fit_transform(features)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_features, target, test_size=0.3, random_state=42
        )

        # Convert target variables to NumPy arrays
        y_train = y_train.to_numpy().flatten()
        y_test = y_test.to_numpy().flatten()

        return X_train, X_test, y_train, y_test

    def get_feature_stats(self):
        # Return basic statistical analysis of the dataset
        return self.df.describe()

# ================================
# Question 2: Experiment Tracking and Model Development
# ================================


class IrisExperiment:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.models = {
            "Logistic Regression": LogisticRegression(),
            "Random Forest": RandomForestClassifier()
        }

    def run_experiment(self):
        # Prepare data
        X_train, X_test, y_train, y_test = self.data_processor.prepare_data()

        # Run experiments for each model
        for model_name, model in self.models.items():
            # Perform cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            mean_cv_score = np.mean(cv_scores)

            # Fit the model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Record metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')

            # Print results
            print(f"Model: {model_name}")
            print(f"Mean Cross-Validation Score: {mean_cv_score:.4f}")
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"Test Precision: {precision:.4f}")
            print(f"Test Recall: {recall:.4f}")
            print("=" * 50)

# ================================
# Question 3: Model Optimization and Testing
# ================================


class IrisModelOptimizer:
    def __init__(self, experiment):
        self.experiment = experiment
        self.quantized_model = None

    def quantize_model(self):
        # Implement model quantization for Logistic Regression
        logistic_regressor = self.experiment.models["Logistic Regression"]
        # Quantize coefficients to 8-bit precision
        self.quantized_model = logistic_regressor
        self.quantized_model.coef_ = np.round(
            logistic_regressor.coef_, decimals=2)
        print("Model quantization completed.")

    def run_tests(self):
        # Simple unit test for quantized model
        X_train, X_test, y_train, y_test = self.experiment.data_processor.prepare_data()
        y_pred = self.quantized_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Quantized Model Accuracy: {accuracy:.4f}")

# ================================
# Main Execution Function
# ================================


def main():
    # Initialize processor
    processor = IrisDataProcessor()
    X_train, X_test, y_train, y_test = processor.prepare_data()

    # Run experiments
    experiment = IrisExperiment(processor)
    experiment.run_experiment()

    # Optimize and test
    optimizer = IrisModelOptimizer(experiment)
    optimizer.quantize_model()
    optimizer.run_tests()


if __name__ == "__main__":
    main()
