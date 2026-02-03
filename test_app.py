from unittest.mock import patch
import requests

APP_URL = "http://localhost:8000"


def get_app_status(url):
    response = requests.get(url)
    return response.status_code


@patch("requests.get")
def test_app_loading(mock_get):
    # mock successful response
    mock_get.return_value.status_code = 200

    status_code = get_app_status(APP_URL)

    assert status_code == 200, "Unable to load Streamlit App"
