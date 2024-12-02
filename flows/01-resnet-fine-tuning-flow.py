from metaflow import FlowSpec, Parameter, step, card, current, project, environment
from common import configure_logging
import logging
import logging.config
import os

configure_logging()
# override the default logging level for the mlflow logger with our own
mlflow_logger = logging.getLogger("mlflow")

@project(name="bird_species_embedding_model")
class FineTuneBirdSpeciesClassifier(FlowSpec):
    """
    Fine-tune a pre-trained model on the bird species dataset.
    """
    TRAIN_DIR = Parameter(
        "train_dir",
        type=str,
        default="../kaggle_data/bird-fifty/train",
        help="The relative location of the training data.",
    )

    TEST_DIR = Parameter(
        "test_dir",
        type=str,
        default="../kaggle_data/bird-fifty/test",
        help="The relative location of the test data.",
    )

    NUMBER_OF_EPOCHS = Parameter(
        "epochs",
        type=int,
        default=15,
        help="The number of epochs to train the model from.",
    )

    BATCH_SIZE = Parameter(
        "batch_size",
        type=int,
        default=32,
        help="The size of the batch per epoch.",
    )
    
    LEARNING_RATE = Parameter(
        "learning_rate",
        type=float,
        default=0.00001,
        help="The learning rate to  use.",
    )

    MODEL_NAME = Parameter(
        "model",
        type=str,
        default="resnet50d.ra2_in1k",
        help=
        """
            Pick a model from:
            - [resnet34d.ra2_in1k, resnet50d.ra2_in1k, resnet152d.ra2_in1k,
            mobilenetv3_small_100.lamb_in1k,mobilenetv3_large_100.ra_in1k,
            vit_large_patch16_224.orig_in21k]
        """,
    )
    
    ACCURACY_THRESHOLD = Parameter(
        "accuracy_threshold",
        type=float,
        default=0.7,
        help=
        """
            Minimum accuracy threshold required to register the model at the end of
            the pipeline. The model will not be registered if its accuracy is below
            this threshold.
        """,
    )

    CHECKPOINT_INTERVAL = Parameter(
        "checkpoint_interval",
        type=int,
        default=1,
        help="Interval (in epochs) to save checkpoints.",
    )

    RESUME_CHECKPOINT = Parameter(
        "checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint path and file to resume training from.",
    )
    
    @card
    @environment(
        vars={
            "MLFLOW_TRACKING_URI": os.getenv(
                "MLFLOW_TRACKING_URI",
                "http://127.0.0.1:5000",
            ),
            "AWS_ACCESS_KEY_ID": os.getenv(
                "AWS_ACCESS_KEY_ID",
                "sN4PjijeDW5Jv8gpjjYP",
            ),
            "AWS_SECRET_ACCESS_KEY": os.getenv(
                "AWS_SECRET_ACCESS_KEY",
                "cPlissnZusAOIRGKBP8RuCV4WIoS75AzUSOIbk4U",
            ),
            "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING": os.getenv(
                "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING",
                "true",
            ),
        },
    )
    @step
    def start(self):
        """
        Start the flow.
        """
        import mlflow
        
        if os.path.exists("app.log"):
            os.remove("app.log")
        
        logging.info("Training %s in flow %s", self.MODEL_NAME, current.flow_name)

        self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

        logging.info("MLFLOW_TRACKING_URI: %s", self.mlflow_tracking_uri)
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        try:
            experiment = mlflow.set_experiment(experiment_name=current.flow_name)
            run = mlflow.start_run(
                run_name=current.run_id,
                experiment_id=experiment.experiment_id
            )
            self.mlflow_run_id = run.info.run_id
            self.experiment_id = experiment.experiment_id
            
            logging.info("Artifact URI: %s", run.info.artifact_uri)
            
            self.training_parameters = {}

        except Exception as e:
            message = f"Failed to connect to MLflow server {self.mlflow_tracking_uri}."
            raise RuntimeError(message) from e

        if os.path.exists("app.log"):
            mlflow.log_artifact("app.log")
        self.next(self.train)

    @card
    @step
    def train(self):
        """
        Train the model.
        """
        import torch
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        from torchvision import datasets
        from torch.utils.data import DataLoader
        import torch.optim as optim
        import torch.nn as nn
        import time
        import os
        import mlflow
        from datetime import datetime

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        with mlflow.start_run(
            run_id=self.mlflow_run_id,
            experiment_id=self.experiment_id,
            log_system_metrics=True,
        ):
            time.sleep(15)
            
            logging.info(mlflow.MlflowClient().get_run(self.mlflow_run_id).data)
            mlflow.autolog(log_models=False)

            if not os.path.exists("mlflow/checkpoints"):
                os.makedirs("mlflow/checkpoints")

            train_folder = self.TRAIN_DIR
            test_folder = self.TEST_DIR

            num_of_classes = len(os.listdir(train_folder))

            model_no_fc_ready_to_fine_tune = timm.create_model(
                self.MODEL_NAME, pretrained=True, num_classes=num_of_classes
            )
            resnet_model = model_no_fc_ready_to_fine_tune

            data_config = resolve_data_config({}, model=resnet_model)
            transform = create_transform(**data_config)
            train_dataset = datasets.ImageFolder(root=train_folder, transform=transform)
            test_dataset = datasets.ImageFolder(root=test_folder, transform=transform)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.BATCH_SIZE,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.BATCH_SIZE,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )

            # Yay for Apple Silicon support! (train on Apple M series GPU)
            if torch.backends.mps.is_available():
                device = "mps"
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            logging.info("Using device: %s", device)
            mlflow.log_param("device", device)

            self.training_parameters = {
                "model": self.MODEL_NAME,
                "batch_size": self.BATCH_SIZE,
                "epochs": self.NUMBER_OF_EPOCHS,
                "learning_rate": self.LEARNING_RATE,
                "train_dir": self.TRAIN_DIR,
                "train_dir_samples": len(train_dataset),
                "test_dir": self.TEST_DIR,
                "test_dir_samples": len(test_dataset),
                "num_classes": num_of_classes,
                "checkpoint_interval": self.CHECKPOINT_INTERVAL,
                "resume_checkpoint": self.RESUME_CHECKPOINT,
            }
            
            mlflow.log_params(self.training_parameters)
            logging.info("Training parameters: %s", self.training_parameters)
            mlflow.log_artifact(os.path.basename(__file__))

            resnet_model.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(resnet_model.parameters(), lr=self.LEARNING_RATE)

            start_epoch = 0
            if self.RESUME_CHECKPOINT:
                checkpoint = torch.load(self.RESUME_CHECKPOINT)
                resnet_model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                start_epoch = checkpoint["epoch"] + 1
                logging.info("Resuming training from epoch %d", start_epoch)

            training_start = time.time()
            for epoch in range(start_epoch, self.NUMBER_OF_EPOCHS):
                epoch_start = time.time()
                resnet_model.train()
                running_loss = 0.0
                correct = 0
                total = 0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = resnet_model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                training_loss = running_loss / len(train_loader)
                logging.info(
                    "Epoch [%d/%d], Training Loss: %.5f",
                    epoch + 1, self.NUMBER_OF_EPOCHS, round(training_loss, 5)
                )
                mlflow.log_metric("training_loss", training_loss, step=epoch)

                train_accuracy = 100 * (correct / total)
                logging.info("Training Accuracy: %.5f%%", train_accuracy)
                mlflow.log_metric("training_accuracy", train_accuracy, step=epoch)

                if (epoch + 1) % self.CHECKPOINT_INTERVAL == 0:
                    checkpoint_path = f"mlflow/checkpoints/{self.MODEL_NAME}_checkpoint.pth"
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": resnet_model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                        },
                        checkpoint_path,
                    )

                    logging.info("Checkpoint saved at %s", checkpoint_path)
                    mlflow.log_artifact(checkpoint_path)

                # Evaluate on test set
                resnet_model.eval()
                running_loss = 0.0
                correct = 0
                total = 0
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = resnet_model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        loss = criterion(outputs, labels)
                        running_loss += loss.item()

                test_loss = running_loss / len(test_loader)
                logging.info(
                    "Epoch [%d/%d], Test Loss: %.5f",
                    epoch + 1, self.NUMBER_OF_EPOCHS, round(test_loss, 5)
                )
                mlflow.log_metric("test_loss", test_loss, step=epoch)

                test_accuracy = 100 * correct / total
                logging.info("Test Accuracy: %.5f%%", round(test_accuracy, 5))
                mlflow.log_metric("test_accuracy", test_accuracy, step=epoch)

                epoch_end = time.time()
                elapsed_time = epoch_end - epoch_start

                mlflow.log_metric("epoch_time", elapsed_time, step=epoch)
                logging.info("Epoch (%d/%d) time: %s", epoch + 1, self.NUMBER_OF_EPOCHS, elapsed_time)
                elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
                
                if os.path.exists("app.log"):
                    mlflow.log_artifact("app.log")

            training_end = time.time()
            elapsed_time = training_end - training_start

            mlflow.log_metric("total_training_time", elapsed_time)
            elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            logging.info("Total training time: %s", elapsed_time)
            
            # Remove the classification head, because we just need the output of the embedding model, and not the classification head
            resnet_model.reset_classifier(0)
            model_path = f"{self.MODEL_NAME}_fine_tune_{num_of_classes}_classes_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pth"
            torch.save(resnet_model.state_dict(), model_path)
            mlflow.log_artifact(model_path)
            
            logging.info("Training complete.")
            if os.path.exists("app.log"):
                mlflow.log_artifact("app.log")
        
        self.next(self.end)

    @card
    @step
    def end(self):
        """
        End the flow.
        """
        import mlflow

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        with mlflow.start_run(
            run_id=self.mlflow_run_id,
            experiment_id=self.experiment_id,
        ):
            logging.info("Workflow %s completed", current.flow_name)

            if os.path.exists("app.log"):
                mlflow.log_artifact("app.log")

if __name__ == "__main__":
    FineTuneBirdSpeciesClassifier()
