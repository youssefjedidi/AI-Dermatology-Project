# # tests/test_data_ingestion.py
# import pytest
# import zipfile
# import os
# from pathlib import Path
# from unittest.mock import MagicMock

# # Import the function we want to test
# from src.data_ingestion import download_and_extract_data

# @pytest.fixture
# def mock_kaggle_api(monkeypatch):
#     """
#     A pytest fixture to mock the KaggleApi and the KAGGLE_JSON_PATH check.
#     """
#     # 1. Mock the KaggleApi class to prevent network calls
#     mock_api_instance = MagicMock()
#     mock_api_class = MagicMock(return_value=mock_api_instance)
#     monkeypatch.setattr("src.data_ingestion.KaggleApi", mock_api_class)
    
#     # 2. --- THE FIX ---
#     # Instead of mocking the .exists method of a real Path object,
#     # we replace the entire KAGGLE_JSON_PATH variable with a mock object.
#     mock_path = MagicMock(spec=Path)
#     mock_path.exists.return_value = True # We program its .exists() to return True
#     monkeypatch.setattr("src.data_ingestion.KAGGLE_JSON_PATH", mock_path)
    
#     return mock_api_instance # Return the API mock for tests to use

# def test_download_and_extract_success(mock_kaggle_api, tmp_path, monkeypatch):
#     """
#     Tests the "happy path" where data is downloaded and extracted successfully.
#     """
#     # Arrange
#     download_dir = tmp_path / "download"
#     extract_dir = tmp_path / "extract"
#     fake_zip_path = download_dir / "dermnet.zip"

#     def create_fake_zip(*args, **kwargs):
#         (kwargs.get("path") / "dermnet.zip").touch()
    
#     mock_kaggle_api.dataset_download_files.side_effect = create_fake_zip

#     mock_zipfile = MagicMock()
#     monkeypatch.setattr(zipfile, "ZipFile", mock_zipfile)
#     mock_os_remove = MagicMock()
#     monkeypatch.setattr(os, "remove", mock_os_remove)

#     # Act
#     download_and_extract_data(
#         dataset_id="fake/dataset",
#         download_path=download_dir,
#         extract_path=extract_dir
#     )

#     # Assert
#     mock_kaggle_api.authenticate.assert_called_once()
#     mock_kaggle_api.dataset_download_files.assert_called_once_with(
#         dataset="fake/dataset", path=download_dir, unzip=False
#     )
#     zipfile.ZipFile.assert_called_once_with(fake_zip_path, 'r')
#     os.remove.assert_called_once_with(fake_zip_path)


# def test_skips_if_data_already_extracted(mock_kaggle_api, tmp_path):
#     """
#     Tests that the function correctly skips all operations if the extract
#     directory is not empty.
#     """
#     # Arrange
#     download_dir = tmp_path / "download"
#     extract_dir = tmp_path / "extract"
#     (extract_dir / "already_exists.txt").mkdir(parents=True, exist_ok=True)
#     (extract_dir / "already_exists.txt").touch()

#     # Act
#     download_and_extract_data(
#         dataset_id="fake/dataset",
#         download_path=download_dir,
#         extract_path=extract_dir
#     )

#     # Assert
#     mock_kaggle_api.authenticate.assert_not_called()
#     mock_kaggle_api.dataset_download_files.assert_not_called()


# def test_raises_error_if_no_zip_found(mock_kaggle_api, tmp_path):
#     """
#     Tests that the function raises a FileNotFoundError if the download
#     call finishes but no .zip file is present.
#     """
#     # Arrange
#     download_dir = tmp_path / "download"
#     extract_dir = tmp_path / "extract"
#     mock_kaggle_api.dataset_download_files.return_value = None

#     # Act & Assert
#     with pytest.raises(FileNotFoundError, match="Error: No zip file found after download."):
#         download_and_extract_data(
#             dataset_id="fake/dataset",
#             download_path=download_dir,
#             extract_path=extract_dir
#         )
    
#     mock_kaggle_api.dataset_download_files.assert_called_once()

# tests/test_data_ingestion.py
import pytest
import zipfile
import os
from pathlib import Path
from unittest.mock import MagicMock

# Import the function we want to test
from src.data_ingestion import download_and_extract_data

@pytest.fixture(autouse=True)
def setup_kaggle_env(monkeypatch):
    """
    Set up fake Kaggle credentials using environment variables.
    This fixture runs automatically for all tests.
    """
    monkeypatch.setenv("KAGGLE_USERNAME", "test_user")
    monkeypatch.setenv("KAGGLE_KEY", "fake_key_for_testing")

@pytest.fixture
def mock_kaggle_api(monkeypatch):
    """
    A pytest fixture to mock the KaggleApi and the KAGGLE_JSON_PATH check.
    """
    # 1. Mock the KaggleApi class to prevent network calls
    mock_api_instance = MagicMock()
    mock_api_class = MagicMock(return_value=mock_api_instance)
    monkeypatch.setattr("src.data_ingestion.KaggleApi", mock_api_class)
    
    # 2. Mock the KAGGLE_JSON_PATH to return True for exists()
    mock_path = MagicMock(spec=Path)
    mock_path.exists.return_value = True
    monkeypatch.setattr("src.data_ingestion.KAGGLE_JSON_PATH", mock_path)
    
    return mock_api_instance

def test_download_and_extract_success(mock_kaggle_api, tmp_path, monkeypatch):
    """
    Tests the "happy path" where data is downloaded and extracted successfully.
    """
    # Arrange
    download_dir = tmp_path / "download"
    extract_dir = tmp_path / "extract"
    fake_zip_path = download_dir / "dermnet.zip"

    def create_fake_zip(*args, **kwargs):
        (kwargs.get("path") / "dermnet.zip").touch()
    
    mock_kaggle_api.dataset_download_files.side_effect = create_fake_zip

    mock_zipfile = MagicMock()
    monkeypatch.setattr(zipfile, "ZipFile", mock_zipfile)
    mock_os_remove = MagicMock()
    monkeypatch.setattr(os, "remove", mock_os_remove)

    # Act
    download_and_extract_data(
        dataset_id="fake/dataset",
        download_path=download_dir,
        extract_path=extract_dir
    )

    # Assert
    mock_kaggle_api.authenticate.assert_called_once()
    mock_kaggle_api.dataset_download_files.assert_called_once_with(
        dataset="fake/dataset", path=download_dir, unzip=False
    )
    zipfile.ZipFile.assert_called_once_with(fake_zip_path, 'r')
    os.remove.assert_called_once_with(fake_zip_path)


def test_skips_if_data_already_extracted(mock_kaggle_api, tmp_path):
    """
    Tests that the function correctly skips all operations if the extract
    directory is not empty.
    """
    # Arrange
    download_dir = tmp_path / "download"
    extract_dir = tmp_path / "extract"
    (extract_dir / "already_exists.txt").mkdir(parents=True, exist_ok=True)
    (extract_dir / "already_exists.txt").touch()

    # Act
    download_and_extract_data(
        dataset_id="fake/dataset",
        download_path=download_dir,
        extract_path=extract_dir
    )

    # Assert
    mock_kaggle_api.authenticate.assert_not_called()
    mock_kaggle_api.dataset_download_files.assert_not_called()


def test_raises_error_if_no_zip_found(mock_kaggle_api, tmp_path):
    """
    Tests that the function raises a FileNotFoundError if the download
    call finishes but no .zip file is present.
    """
    # Arrange
    download_dir = tmp_path / "download"
    extract_dir = tmp_path / "extract"
    mock_kaggle_api.dataset_download_files.return_value = None

    # Act & Assert
    with pytest.raises(FileNotFoundError, match="Error: No zip file found after download."):
        download_and_extract_data(
            dataset_id="fake/dataset",
            download_path=download_dir,
            extract_path=extract_dir
        )
    
    mock_kaggle_api.dataset_download_files.assert_called_once()

def test_missing_kaggle_credentials(monkeypatch, tmp_path):
    """
    Tests that the function raises a FileNotFoundError when kaggle.json is missing.
    """
    # Remove the environment variables for this test
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    
    # Mock KAGGLE_JSON_PATH to not exist
    mock_path = MagicMock(spec=Path)
    mock_path.exists.return_value = False
    mock_path.__str__ = MagicMock(return_value="/fake/path/kaggle.json")
    monkeypatch.setattr("src.data_ingestion.KAGGLE_JSON_PATH", mock_path)
    
    download_dir = tmp_path / "download"
    extract_dir = tmp_path / "extract"

    # Act & Assert
    with pytest.raises(FileNotFoundError, match="Kaggle API credentials not found"):
        download_and_extract_data(
            dataset_id="fake/dataset",
            download_path=download_dir,
            extract_path=extract_dir
        )