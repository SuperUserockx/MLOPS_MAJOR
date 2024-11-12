import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score

class IrisExperiment:
    def __init__(self, data_processor):
        # Initialize your experiment
        self.data_processor = data_processor
        self.models = {
            "Logistic Regression": LogisticRegression(),
            "Random Forest": RandomForestClassifier()
        }
        mlflow.set_experiment("Iris Classification Experiment")

    def run_experiment(self):
        # Implement experiment workflow
        for model_name, model in self.models.items():
            with mlflow.start_run(run_name=model_name):
                model.fit(self.data_processor.X_train, self.data_processor.y_train)
                y_pred = model.predict(self.data_processor.X_test)

                # Record metrics
                accuracy = accuracy_score(self.data_processor.y_test, y_pred)
                precision = precision_score(self.data_processor.y_test, y_pred, average='weighted')
                recall = recall_score(self.data_processor.y_test, y_pred, average='weighted')
                
                # Log results in MLflow
                self.log_results(model_name, accuracy, precision, recall)
                
                print(f"{model_name} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}")

    def log_results(self, model_name, accuracy, precision, recall):
        # Implement MLflow logging
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        input_example = self.data_processor.X_test[:5]  # Using a few samples as input example
        mlflow.sklearn.log_model(self.models[model_name], model_name, input_example=input_example)
        print(f"Results logged in MLflow for {model_name}.")

# Instantiate and run
experiment = IrisExperiment(processor)
experiment.run_experiment()
