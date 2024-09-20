# Sales Forecasting Tool

This repository contains a Streamlit-based application for sales forecasting. The tool allows users to upload sales data, process it, visualize trends, and generate forecasts using various models.

## Features

- Data loading and processing
- Data visualization with interactive charts
- Forecasting using multiple models:
  - Prophet
  - SARIMA
  - Linear Regression
  - Dummy (baseline) model
- Model comparison and selection
- Forecast visualization and export

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/Quentin-DS/Project-Forecast.git
   cd sales-forecasting-tool
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run streamlit_app.py
   ```

2. Open your web browser and go to the URL displayed in the terminal (usually `http://localhost:8501`).

3. Follow the instructions in the app to upload your data, process it, and generate forecasts.

## File Structure

- `streamlit_app.py`: Main Streamlit application file
- `forecast_functions.py`: Contains functions for various forecasting models and utilities
- `requirements.txt`: List of Python packages required to run the application
- `README.md`: This file, containing information about the project

## Data Format

The app expects an Excel file with the following columns:
- `ds`: Date column (format: YYYY-MM-DD)
- `y`: Sales values
- `item`: Item identifier

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
