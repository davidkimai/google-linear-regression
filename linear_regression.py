#!/usr/bin/env python3
"""
Enterprise Linear Regression Framework
A minimal, dataset-agnostic implementation for training, evaluating, and deploying linear regression models.

Design Principles (Zen of Python):
- Simple is better than complex
- Explicit is better than implicit  
- Flat is better than nested
- Readability counts
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Callable
from datetime import datetime

import numpy as np
import pandas as pd
import keras
from sklearn.model_selection import train_test_split


# ============================================================================
# Configuration - Explicit specification of all experiment parameters
# ============================================================================

@dataclass
class DatasetConfig:
    """Dataset configuration - where and how to load data."""
    csv_path: str
    features: List[str]
    label: str
    test_size: float = 0.2
    random_state: int = 42
    transforms: Dict[str, str] = field(default_factory=dict)  # {'new_col': 'expression'}


@dataclass
class ModelConfig:
    """Model architecture and training configuration."""
    learning_rate: float = 0.001
    epochs: int = 20
    batch_size: int = 50
    optimizer: str = 'rmsprop'  # rmsprop, adam, sgd
    validation_split: float = 0.2


@dataclass
class ExperimentConfig:
    """Complete experiment specification."""
    name: str
    dataset: DatasetConfig
    model: ModelConfig
    
    def save(self, path: Path) -> None:
        """Save configuration to JSON file."""
        path.write_text(json.dumps(asdict(self), indent=2))
    
    @classmethod
    def load(cls, path: Path) -> 'ExperimentConfig':
        """Load configuration from JSON file."""
        data = json.loads(path.read_text())
        return cls(
            name=data['name'],
            dataset=DatasetConfig(**data['dataset']),
            model=ModelConfig(**data['model'])
        )


# ============================================================================
# Data Pipeline - Load, transform, and prepare datasets
# ============================================================================

class RegressionDataset:
    """Dataset abstraction for any tabular regression problem."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.df: Optional[pd.DataFrame] = None
        self.X_train: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Configure structured logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load(self) -> 'RegressionDataset':
        """Load CSV data with validation."""
        try:
            self.df = pd.read_csv(self.config.csv_path)
            self.logger.info(f"Loaded dataset: {len(self.df)} rows, {len(self.df.columns)} columns")
            self._validate_schema()
            return self
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise
    
    def _validate_schema(self) -> None:
        """Validate dataset contains required features and label."""
        missing = set(self.config.features + [self.config.label]) - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}. Available: {list(self.df.columns)}")
        
        # Check for missing values
        null_counts = self.df[self.config.features + [self.config.label]].isnull().sum()
        if null_counts.any():
            self.logger.warning(f"Missing values detected:\n{null_counts[null_counts > 0]}")
    
    def transform(self) -> 'RegressionDataset':
        """Apply feature transformations specified in config."""
        for new_col, expression in self.config.transforms.items():
            try:
                self.df[new_col] = self.df.eval(expression)
                self.logger.info(f"Created feature '{new_col}' = {expression}")
            except Exception as e:
                self.logger.error(f"Transform failed for '{new_col}': {e}")
                raise
        return self
    
    def split(self) -> 'RegressionDataset':
        """Split into train/test sets with stratification awareness."""
        X = self.df[self.config.features].values
        y = self.df[self.config.label].values
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )
        
        self.logger.info(
            f"Split dataset: train={len(self.X_train)}, test={len(self.X_test)}"
        )
        return self
    
    def describe(self) -> pd.DataFrame:
        """Return descriptive statistics for dataset."""
        return self.df[self.config.features + [self.config.label]].describe()
    
    def correlation_matrix(self) -> pd.DataFrame:
        """Compute correlation matrix for features and label."""
        return self.df[self.config.features + [self.config.label]].corr()


# ============================================================================
# Model - Linear regression with flexible architecture
# ============================================================================

class LinearRegressionModel:
    """Keras-based linear regression with automatic feature handling."""
    
    def __init__(self, config: ModelConfig, n_features: int, feature_names: List[str]):
        self.config = config
        self.n_features = n_features
        self.feature_names = feature_names
        self.model: Optional[keras.Model] = None
        self.history: Optional[Dict[str, List[float]]] = None
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def build(self) -> 'LinearRegressionModel':
        """Construct linear regression model architecture."""
        # Input layer for each feature
        inputs = [keras.Input(shape=(1,), name=f'input_{i}') for i in range(self.n_features)]
        
        # Concatenate all inputs
        if len(inputs) > 1:
            concatenated = keras.layers.Concatenate()(inputs)
        else:
            concatenated = inputs[0]
        
        # Single dense layer for linear regression
        output = keras.layers.Dense(units=1, name='output')(concatenated)
        
        self.model = keras.Model(inputs=inputs, outputs=output)
        
        # Compile with specified optimizer
        optimizer = self._get_optimizer()
        self.model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=[keras.metrics.RootMeanSquaredError(name='rmse'),
                    keras.metrics.MeanAbsoluteError(name='mae')]
        )
        
        self.logger.info(f"Built model: {self.n_features} features → 1 output")
        return self
    
    def _get_optimizer(self) -> keras.optimizers.Optimizer:
        """Get configured optimizer instance."""
        optimizers = {
            'rmsprop': keras.optimizers.RMSprop,
            'adam': keras.optimizers.Adam,
            'sgd': keras.optimizers.SGD
        }
        
        optimizer_class = optimizers.get(self.config.optimizer.lower())
        if not optimizer_class:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        return optimizer_class(learning_rate=self.config.learning_rate)
    
    def train(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegressionModel':
        """Train model on dataset."""
        if self.model is None:
            raise RuntimeError("Model not built. Call build() first.")
        
        # Split features for multi-input model
        X_split = [X[:, i:i+1] for i in range(self.n_features)]
        
        history = self.model.fit(
            X_split, y,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_split=self.config.validation_split,
            verbose=0
        )
        
        self.history = history.history
        final_rmse = self.history['rmse'][-1]
        final_val_rmse = self.history['val_rmse'][-1]
        
        self.logger.info(
            f"Training complete: RMSE={final_rmse:.4f}, Val_RMSE={final_val_rmse:.4f}"
        )
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions for input features."""
        if self.model is None:
            raise RuntimeError("Model not built/loaded. Call build() or load() first.")
        
        X_split = [X[:, i:i+1] for i in range(self.n_features)]
        return self.model.predict(X_split, verbose=0).flatten()
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model on test set."""
        if self.model is None:
            raise RuntimeError("Model not built/loaded. Call build() or load() first.")
        
        X_split = [X[:, i:i+1] for i in range(self.n_features)]
        results = self.model.evaluate(X_split, y, verbose=0, return_dict=True)
        
        self.logger.info(f"Evaluation: {results}")
        return results
    
    def save(self, path: Path, metadata: Dict[str, Any]) -> None:
        """Save model with comprehensive metadata."""
        if self.model is None:
            raise RuntimeError("No model to save.")
        
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save Keras model
        self.model.save(save_dir / 'model.keras')
        
        # Save metadata
        metadata_full = {
            **metadata,
            'model_config': asdict(self.config),
            'feature_names': self.feature_names,
            'n_features': self.n_features,
            'history': self.history,
            'saved_at': datetime.now().isoformat(),
            'keras_version': keras.__version__,
        }
        
        (save_dir / 'metadata.json').write_text(
            json.dumps(metadata_full, indent=2)
        )
        
        self.logger.info(f"Saved model to {save_dir}")
    
    @classmethod
    def load(cls, path: Path) -> 'LinearRegressionModel':
        """Load model with metadata."""
        load_dir = Path(path)
        
        # Load metadata
        metadata = json.loads((load_dir / 'metadata.json').read_text())
        
        # Reconstruct config
        config = ModelConfig(**metadata['model_config'])
        
        # Create instance
        instance = cls(
            config=config,
            n_features=metadata['n_features'],
            feature_names=metadata['feature_names']
        )
        
        # Load Keras model
        instance.model = keras.models.load_model(load_dir / 'model.keras')
        instance.history = metadata.get('history')
        
        instance.logger.info(f"Loaded model from {load_dir}")
        return instance
    
    def get_coefficients(self) -> Dict[str, float]:
        """Extract learned coefficients (weights) from model."""
        if self.model is None:
            raise RuntimeError("Model not built/loaded.")
        
        # Get weights from dense layer
        weights = self.model.get_layer('output').get_weights()[0].flatten()
        bias = self.model.get_layer('output').get_weights()[1][0]
        
        coef_dict = {name: float(w) for name, w in zip(self.feature_names, weights)}
        coef_dict['bias'] = float(bias)
        
        return coef_dict


# ============================================================================
# Experiment Runner - Orchestrate full ML lifecycle
# ============================================================================

class ExperimentRunner:
    """Orchestrate dataset loading, training, evaluation, and persistence."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.dataset: Optional[RegressionDataset] = None
        self.model: Optional[LinearRegressionModel] = None
        self.results: Dict[str, Any] = {}
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def run(self) -> 'ExperimentRunner':
        """Execute complete experiment workflow."""
        self.logger.info(f"Starting experiment: {self.config.name}")
        
        # Data pipeline
        self.dataset = (
            RegressionDataset(self.config.dataset)
            .load()
            .transform()
            .split()
        )
        
        # Model pipeline
        self.model = (
            LinearRegressionModel(
                config=self.config.model,
                n_features=len(self.config.dataset.features),
                feature_names=self.config.dataset.features
            )
            .build()
            .train(self.dataset.X_train, self.dataset.y_train)
        )
        
        # Evaluation
        train_metrics = self.model.evaluate(self.dataset.X_train, self.dataset.y_train)
        test_metrics = self.model.evaluate(self.dataset.X_test, self.dataset.y_test)
        
        self.results = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'coefficients': self.model.get_coefficients(),
            'dataset_stats': self.dataset.describe().to_dict(),
            'correlation_matrix': self.dataset.correlation_matrix().to_dict()
        }
        
        self.logger.info(f"Experiment complete: Test RMSE = {test_metrics['rmse']:.4f}")
        return self
    
    def save(self, output_dir: Path) -> None:
        """Save experiment artifacts."""
        output_path = Path(output_dir) / self.config.name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        self.config.save(output_path / 'config.json')
        
        # Save model
        self.model.save(
            output_path / 'model',
            metadata={
                'experiment_name': self.config.name,
                'dataset_config': asdict(self.config.dataset),
            }
        )
        
        # Save results
        (output_path / 'results.json').write_text(
            json.dumps(self.results, indent=2)
        )
        
        self.logger.info(f"Saved experiment to {output_path}")
    
    @classmethod
    def load(cls, experiment_dir: Path) -> 'ExperimentRunner':
        """Load complete experiment from disk."""
        exp_path = Path(experiment_dir)
        
        # Load config
        config = ExperimentConfig.load(exp_path / 'config.json')
        
        # Create instance
        instance = cls(config)
        
        # Load model
        instance.model = LinearRegressionModel.load(exp_path / 'model')
        
        # Load results
        if (exp_path / 'results.json').exists():
            instance.results = json.loads((exp_path / 'results.json').read_text())
        
        return instance
    
    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions for batch of inputs."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call run() or load() first.")
        return self.model.predict(X)


# ============================================================================
# CLI Interface - Command-line access to framework
# ============================================================================

def create_example_config(output_path: Path, dataset_type: str = 'taxi') -> None:
    """Generate example configuration file for quick start."""
    
    configs = {
        'taxi': ExperimentConfig(
            name='chicago_taxi_regression',
            dataset=DatasetConfig(
                csv_path='https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv',
                features=['TRIP_MILES', 'TRIP_MINUTES'],
                label='FARE',
                transforms={'TRIP_MINUTES': 'TRIP_SECONDS / 60'}
            ),
            model=ModelConfig(
                learning_rate=0.001,
                epochs=20,
                batch_size=50
            )
        ),
        'housing': ExperimentConfig(
            name='housing_price_regression',
            dataset=DatasetConfig(
                csv_path='path/to/housing.csv',
                features=['square_feet', 'bedrooms', 'age'],
                label='price'
            ),
            model=ModelConfig()
        )
    }
    
    config = configs.get(dataset_type, configs['taxi'])
    config.save(output_path)
    print(f"Created example config: {output_path}")


def train_command(args: argparse.Namespace) -> None:
    """Execute training from configuration file."""
    config = ExperimentConfig.load(Path(args.config))
    
    runner = ExperimentRunner(config).run()
    
    if args.output:
        runner.save(Path(args.output))
    
    # Print summary
    print("\n" + "="*80)
    print(f"EXPERIMENT: {config.name}")
    print("="*80)
    print(f"\nTest RMSE: {runner.results['test_metrics']['rmse']:.4f}")
    print(f"Test MAE:  {runner.results['test_metrics']['mae']:.4f}")
    print(f"\nLearned Coefficients:")
    for name, coef in runner.results['coefficients'].items():
        print(f"  {name:20s}: {coef:10.4f}")


def predict_command(args: argparse.Namespace) -> None:
    """Generate predictions using trained model."""
    runner = ExperimentRunner.load(Path(args.experiment_dir))
    
    # Load input data
    input_df = pd.read_csv(args.input_csv)
    X = input_df[runner.config.dataset.features].values
    
    predictions = runner.predict_batch(X)
    
    # Save predictions
    output_df = input_df.copy()
    output_df['prediction'] = predictions
    output_df.to_csv(args.output_csv, index=False)
    
    print(f"Saved predictions to {args.output_csv}")


def evaluate_command(args: argparse.Namespace) -> None:
    """Evaluate model on new dataset."""
    runner = ExperimentRunner.load(Path(args.experiment_dir))
    
    # Load evaluation data
    eval_df = pd.read_csv(args.eval_csv)
    X = eval_df[runner.config.dataset.features].values
    y = eval_df[runner.config.dataset.label].values
    
    metrics = runner.model.evaluate(X, y)
    
    print("\n" + "="*80)
    print(f"EVALUATION RESULTS")
    print("="*80)
    for metric_name, value in metrics.items():
        print(f"{metric_name:20s}: {value:.4f}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Enterprise Linear Regression Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create config command
    config_parser = subparsers.add_parser('create-config', help='Generate example configuration')
    config_parser.add_argument('--output', required=True, help='Output path for config file')
    config_parser.add_argument('--type', default='taxi', choices=['taxi', 'housing'],
                              help='Type of example config')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train model from config')
    train_parser.add_argument('--config', required=True, help='Path to experiment config JSON')
    train_parser.add_argument('--output', help='Output directory for experiment artifacts')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Generate predictions')
    predict_parser.add_argument('--experiment-dir', required=True, help='Path to experiment directory')
    predict_parser.add_argument('--input-csv', required=True, help='Input CSV with features')
    predict_parser.add_argument('--output-csv', required=True, help='Output CSV with predictions')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model on new data')
    eval_parser.add_argument('--experiment-dir', required=True, help='Path to experiment directory')
    eval_parser.add_argument('--eval-csv', required=True, help='CSV with features and labels')
    
    args = parser.parse_args()
    
    if args.command == 'create-config':
        create_example_config(Path(args.output), args.type)
    elif args.command == 'train':
        train_command(args)
    elif args.command == 'predict':
        predict_command(args)
    elif args.command == 'evaluate':
        evaluate_command(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
