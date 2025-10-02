# Enterprise Linear Regression Framework

A minimal, dataset-agnostic implementation for training, evaluating, and deploying linear regression models. Built for enterprise production with configuration-driven experiments, comprehensive model persistence, and dual API/CLI interfaces.

**Design Philosophy:** Maximal signal, minimal complexity. Single-file implementation (~450 LOC) following Zen of Python principles.

## Features

- **Dataset Agnostic:** Works with any tabular CSV regression problem (taxi fares, housing prices, sales forecasting)
- **Configuration Driven:** Reproducible experiments via JSON configs with full parameter tracking
- **Production Ready:** Comprehensive logging, error handling, model serialization with metadata
- **Dual Interface:** Both programmatic API and CLI for diverse workflows
- **Minimal Dependencies:** Core ML stack only (Keras, TensorFlow, pandas, NumPy, scikit-learn)

## Installation

```bash
git clone https://github.com/davidkimai/google-linear-regression.git
cd google-linear-regression
pip install -r requirements.txt
```

**Requirements:** Python 3.9+

## Quick Start

### 1. Generate Example Configuration

```bash
python linear_regression.py create-config --output taxi_config.json --type taxi
```

### 2. Train Model

```bash
python linear_regression.py train --config taxi_config.json --output experiments/
```

### 3. Evaluate Results

```bash
python linear_regression.py evaluate \
    --experiment-dir experiments/chicago_taxi_regression \
    --eval-csv test_data.csv
```

### 4. Generate Predictions

```bash
python linear_regression.py predict \
    --experiment-dir experiments/chicago_taxi_regression \
    --input-csv new_trips.csv \
    --output-csv predictions.csv
```

## Programmatic API

### Training a Model

```python
from linear_regression import ExperimentConfig, DatasetConfig, ModelConfig, ExperimentRunner
from pathlib import Path

# Define experiment configuration
config = ExperimentConfig(
    name='taxi_fare_prediction',
    dataset=DatasetConfig(
        csv_path='https://download.mlcc.google.com/mledu-datasets/chicago_taxi_train.csv',
        features=['TRIP_MILES', 'TRIP_MINUTES'],
        label='FARE',
        transforms={'TRIP_MINUTES': 'TRIP_SECONDS / 60'}  # Feature engineering
    ),
    model=ModelConfig(
        learning_rate=0.001,
        epochs=20,
        batch_size=50,
        optimizer='rmsprop'
    )
)

# Run experiment
runner = ExperimentRunner(config).run()

# Save artifacts
runner.save(Path('experiments/'))

# Inspect results
print(f"Test RMSE: {runner.results['test_metrics']['rmse']:.4f}")
print(f"Coefficients: {runner.results['coefficients']}")
```

### Making Predictions

```python
from linear_regression import ExperimentRunner
import numpy as np

# Load trained model
runner = ExperimentRunner.load(Path('experiments/taxi_fare_prediction'))

# Prepare input features
X = np.array([[5.2, 15.3], [10.8, 28.7]])  # [TRIP_MILES, TRIP_MINUTES]

# Generate predictions
predictions = runner.predict_batch(X)
print(predictions)  # [predicted_fare_1, predicted_fare_2]
```

### Inspecting Dataset Statistics

```python
from linear_regression import RegressionDataset, DatasetConfig

config = DatasetConfig(
    csv_path='data.csv',
    features=['feature1', 'feature2'],
    label='target'
)

dataset = RegressionDataset(config).load()

# Descriptive statistics
print(dataset.describe())

# Feature correlations
print(dataset.correlation_matrix())
```

## Configuration Reference

### DatasetConfig

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `csv_path` | str | Path or URL to CSV file | Required |
| `features` | List[str] | Feature column names | Required |
| `label` | str | Target column name | Required |
| `test_size` | float | Train/test split ratio | 0.2 |
| `random_state` | int | Random seed for reproducibility | 42 |
| `transforms` | Dict[str, str] | Feature engineering expressions | {} |

### ModelConfig

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `learning_rate` | float | Optimizer learning rate | 0.001 |
| `epochs` | int | Training epochs | 20 |
| `batch_size` | int | Batch size for training | 50 |
| `optimizer` | str | Optimizer (rmsprop, adam, sgd) | rmsprop |
| `validation_split` | float | Validation data ratio | 0.2 |

### Feature Transforms

Transforms use pandas `.eval()` syntax for feature engineering:

```json
{
  "transforms": {
    "TRIP_MINUTES": "TRIP_SECONDS / 60",
    "PRICE_PER_MILE": "FARE / TRIP_MILES",
    "LOG_DISTANCE": "log(TRIP_MILES + 1)"
  }
}
```

## Multi-Dataset Examples

### Housing Price Prediction

```python
config = ExperimentConfig(
    name='housing_prices',
    dataset=DatasetConfig(
        csv_path='housing.csv',
        features=['square_feet', 'bedrooms', 'age_years'],
        label='sale_price',
        transforms={'price_per_sqft': 'sale_price / square_feet'}
    ),
    model=ModelConfig(learning_rate=0.01, epochs=50)
)

ExperimentRunner(config).run().save(Path('experiments/'))
```

### Sales Forecasting

```python
config = ExperimentConfig(
    name='sales_forecast',
    dataset=DatasetConfig(
        csv_path='sales_data.csv',
        features=['advertising_spend', 'seasonality_index', 'competitor_price'],
        label='monthly_revenue'
    ),
    model=ModelConfig(optimizer='adam', batch_size=100)
)

ExperimentRunner(config).run().save(Path('experiments/'))
```

## Model Persistence

Every experiment saves:
- **model.keras:** Trained Keras model
- **metadata.json:** Full experiment metadata (config, features, metrics, timestamps, versions)
- **config.json:** Experiment configuration for reproducibility
- **results.json:** Training/test metrics, coefficients, dataset statistics

### Loading Saved Models

```python
runner = ExperimentRunner.load(Path('experiments/my_experiment'))

# Access model
predictions = runner.model.predict(X)

# Access metadata
coefficients = runner.results['coefficients']
test_rmse = runner.results['test_metrics']['rmse']
```

## Experiment Comparison

Compare multiple experiments programmatically:

```python
experiments = ['exp1', 'exp2', 'exp3']
results = []

for exp_name in experiments:
    runner = ExperimentRunner.load(Path(f'experiments/{exp_name}'))
    results.append({
        'name': exp_name,
        'test_rmse': runner.results['test_metrics']['rmse'],
        'coefficients': runner.results['coefficients']
    })

# Find best model
best = min(results, key=lambda x: x['test_rmse'])
print(f"Best model: {best['name']} (RMSE: {best['test_rmse']:.4f})")
```

## Hyperparameter Tuning

Grid search example:

```python
learning_rates = [0.0001, 0.001, 0.01]
batch_sizes = [32, 50, 100]

best_rmse = float('inf')
best_config = None

for lr in learning_rates:
    for bs in batch_sizes:
        config = ExperimentConfig(
            name=f'tune_lr{lr}_bs{bs}',
            dataset=base_dataset_config,
            model=ModelConfig(learning_rate=lr, batch_size=bs)
        )
        
        runner = ExperimentRunner(config).run()
        test_rmse = runner.results['test_metrics']['rmse']
        
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            best_config = config
            
print(f"Best hyperparameters: LR={best_config.model.learning_rate}, BS={best_config.model.batch_size}")
```

## Validation Ground Truth (Taxi Dataset)

Chicago taxi fares follow this formula:
```
FARE = 2.25 * TRIP_MILES + 0.12 * TRIP_MINUTES + 3.25
```

Compare learned coefficients against ground truth:

```python
coefficients = runner.results['coefficients']
print(f"Learned: {coefficients['TRIP_MILES']:.2f} (Ground truth: 2.25)")
print(f"Learned: {coefficients['TRIP_MINUTES']:.2f} (Ground truth: 0.12)")
print(f"Learned: {coefficients['bias']:.2f} (Ground truth: 3.25)")
```

Expected RMSE with `TRIP_MILES` + `TRIP_MINUTES`: ~3.5

## Architecture

**Single-file design** (`linear_regression.py`) with clear component separation:

```
Configuration Layer (Dataclasses)
├── DatasetConfig: CSV path, features, transforms
├── ModelConfig: Hyperparameters, optimizer
└── ExperimentConfig: Complete experiment specification

Data Pipeline (RegressionDataset)
├── Load: CSV ingestion with validation
├── Transform: Feature engineering
├── Split: Train/test partitioning
└── Analyze: Statistics and correlations

Model Layer (LinearRegressionModel)
├── Build: Keras model construction
├── Train: Fit with history tracking
├── Predict: Inference on new data
├── Evaluate: Metrics computation
└── Persist: Save/load with metadata

Experiment Runner (ExperimentRunner)
├── Orchestrate: Full ML lifecycle
├── Track: Comprehensive metrics
└── Persist: Reproducible artifacts

CLI Interface
├── create-config: Generate example configs
├── train: Execute experiments
├── predict: Batch inference
└── evaluate: Model assessment
```

## Logging

Structured logging for production observability:

```python
2024-10-01 12:00:00 - RegressionDataset - INFO - Loaded dataset: 50000 rows, 6 columns
2024-10-01 12:00:01 - RegressionDataset - INFO - Created feature 'TRIP_MINUTES' = TRIP_SECONDS / 60
2024-10-01 12:00:02 - RegressionDataset - INFO - Split dataset: train=40000, test=10000
2024-10-01 12:00:03 - LinearRegressionModel - INFO - Built model: 2 features → 1 output
2024-10-01 12:00:15 - LinearRegressionModel - INFO - Training complete: RMSE=3.4787, Val_RMSE=3.5123
2024-10-01 12:00:16 - ExperimentRunner - INFO - Experiment complete: Test RMSE = 3.5089
```

## Error Handling

Comprehensive validation with actionable messages:

```python
# Missing features
ValueError: Missing columns: {'TRIP_MILES'}. Available: ['FARE', 'TRIP_SECONDS', 'COMPANY']

# Invalid optimizer
ValueError: Unknown optimizer: sgdd. Choose from: rmsprop, adam, sgd

# Transform errors
SyntaxError: Transform failed for 'TRIP_MINUTES': name 'TRIP_SECOND' is not defined
```

## Troubleshooting

**Import errors:** Ensure all dependencies installed: `pip install -r requirements.txt`

**CSV loading fails:** Check network connectivity for URLs, file permissions for local paths

**High RMSE:** Verify feature scaling (very large/small values), check for missing values, increase epochs

**Training divergence:** Reduce learning rate (e.g., 0.0001), decrease batch size, normalize features

**Memory errors:** Reduce batch size, use chunked CSV reading for large datasets

## Contributing

This is a minimal reference implementation from Google ML Crash Course. For production deployments:
- Add unit tests (pytest framework)
- Implement cross-validation
- Add regularization (L1/L2)
- Support categorical features (one-hot encoding)
- Integrate experiment tracking (MLflow, W&B)
- Containerize with Docker

## License

Apache License 2.0 (consistent with source material from Google ML Crash Course)

## Citation

Original Colab: [Linear Regression Taxi](https://colab.research.google.com/github/google/eng-edu/blob/main/ml/cc/exercises/linear_regression_taxi.ipynb)

```bibtex
@misc{google-ml-crash-course,
  title={Machine Learning Crash Course - Linear Regression},
  author={Google},
  year={2023},
  url={https://developers.google.com/machine-learning/crash-course}
}
```
