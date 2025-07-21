import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys


project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from validator.validation_runner import ValidationRunner
from validator.modules.onnx import (
    ONNXValidationModule,
    ONNXConfig,
    ONNXInputData,
    ONNXMetrics,
)


def create_test_csv_data():
    """Create simple test CSV data with target column included"""

    np.random.seed(42)

    test_data = {
        "feature1": np.random.normal(0, 1, 20),
        "feature2": np.random.normal(0, 1, 20),
        "feature3": np.random.normal(0, 1, 20),
        "target": np.random.normal(0, 0.5, 20),  # Include target in the same dataset
    }
    test_df = pd.DataFrame(test_data)
    test_csv = test_df.to_csv(index=False)

    return test_csv


@patch("validator.validation_runner.FedLedger")
@patch("validator.modules.onnx.hf_hub_download")
@patch("validator.modules.onnx.ort.InferenceSession")
@patch("requests.get")
def test_onnx_validation_works(
    mock_requests, mock_ort, mock_hf_download, mock_fedledger
):
    """Test that ONNX validation can complete successfully"""

    test_csv = create_test_csv_data()

    # Mock API
    mock_api = MagicMock()
    mock_api.list_tasks.return_value = [
        {"id": 1, "task_type": "onnx", "title": "Test", "data": {}}
    ]
    mock_api.mark_assignment_as_failed = MagicMock()
    mock_fedledger.return_value = mock_api

    mock_hf_download.return_value = "/user-name/model.onnx"

    mock_session = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "input"
    mock_session.get_inputs.return_value = [mock_input]

    # Generate 20 predictions to match the ground truth
    np.random.seed(123)
    fake_predictions = np.random.normal(0, 0.3, 20)

    mock_session.run.return_value = [fake_predictions.reshape(-1, 1)]
    mock_ort.return_value = mock_session

    # Mock HTTP requests for CSV data
    def mock_get_side_effect(url):
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.text = test_csv  # Same CSV contains both features and target
        return response

    mock_requests.side_effect = mock_get_side_effect

    runner = ValidationRunner(
        module="onnx",
        task_ids=[1],
        flock_api_key="test_key",
        hf_token="test_token",
        test_mode=True,
    )

    input_data = ONNXInputData(
        model_repo_id="test/model",
        revision="main",
        test_data_url="https://example.com/test.csv",
        target_column="target",
        task_type="forecasting",
        task_id=1,
        required_metrics=[
            "mae",
            "rmse",
            "mape",
            "smape",
            "mse",
            "r2_score",
            "directional_accuracy",
            "forecast_skill",
        ],
    )

    # Perform validation
    print("Running ONNX validation...")
    metrics = runner.perform_validation("assignment_123", 1, input_data)

    print(f"Validation result: {metrics}")

    if metrics is None:
        print("Validation returned None - something went wrong")
        # Let's check what might have failed
        print("Checking mocks:")
        print(f"  - HuggingFace download called: {mock_hf_download.called}")
        print(f"  - ONNX Runtime called: {mock_ort.called}")
        print(f"  - HTTP requests called: {mock_requests.call_count}")
        return False
    else:
        print("Validation completed successfully!")
        print(f"   - Type: {type(metrics)}")
        if hasattr(metrics, "mae"):
            print(f"   - MAE: {metrics.mae}")
        if hasattr(metrics, "rmse"):
            print(f"   - RMSE: {metrics.rmse}")
        return True


if __name__ == "__main__":
    print("Testing ONNX Module")
    print("=" * 50)

    # Run tests
    print()
    test_passed = test_onnx_validation_works()

    if test_passed:
        print("\nAll ONNX tests passed!")
        sys.exit(0)
    else:
        print("\nSome tests failed")
        sys.exit(1)
