# NYC Crime and Risk Prediction

This repository contains the implementation of the NYC Crime and Risk Prediction system. The project uses machine learning models and data visualization tools to analyze and predict crime patterns in New York City.

## Features

- **Data Preprocessing and Analysis**: Cleaned and analyzed NYC crime data to identify trends and correlations. Visualizations created using Matplotlib and Seaborn provide insights into crime patterns.
- **Crime Prediction**: Implemented machine learning models to predict crime types and risk levels in various zones.
- **Risk Visualization**: Created a Tableau dashboard to visualize crime trends and risk levels.
- **Route Prediction**: Designed a system to predict safe routes based on crime data.

## Files

- **AJU_BIGDATA_TABLEAU_DASHBOARD.twb**: Tableau dashboard file for crime and risk visualization.
- **AJU_FinalProject_Presentation_SP24_BigData.pptx**: Final project presentation.
- **AJU_MODEL_CRIME_RISK_PART[1-7].pkl**: Crime risk prediction model files.
- **AJU_MODEL_CRIME_TYPE.pkl**: Crime type prediction model file.
- **CRIME_PREDICTION.ipynb**: Jupyter notebook for crime prediction.
- **DATA_PREPROCESSING_AND_ANALYSIS.ipynb**: Jupyter notebook for data preprocessing and analysis.
- **ROUTE_PREDICTION.ipynb**: Jupyter notebook for route prediction.
- **ZONE_RISK.csv**: Dataset containing zone-wise crime risk levels.
- **PROJECT.py**: Python script for integrating the crime prediction workflow.
- **requirements.txt**: Python dependencies required for the project.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/amarnadh145/NYC_CRIME_AND_RISK_PREDICTION.git
   cd NYC_CRIME_AND_RISK_PREDICTION
   ```

2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install Tableau to open the `.twb` file for visualization.

## Usage

1. **Run Data Preprocessing**:
   Open and run `DATA_PREPROCESSING_AND_ANALYSIS.ipynb` in a Jupyter notebook to preprocess and analyze the data. This step cleans the dataset, performs exploratory data analysis (EDA), and generates visualizations using Matplotlib and Seaborn. The preprocessed dataset, available via a [Google Drive link](https://drive.google.com/), is used in subsequent notebooks.

2. **Train and Test Models**:
   Use `CRIME_PREDICTION.ipynb` to train machine learning models for predicting crime types and risk levels.

3. **Safe Route Prediction**:
   Run `ROUTE_PREDICTION.ipynb` to predict safe routes.

4. **Visualization**:
   Open `AJU_BIGDATA_TABLEAU_DASHBOARD.twb` in Tableau to explore crime trends and risk levels.

## Project Presentation

Refer to `AJU_FinalProject_Presentation_SP24_BigData.pptx` for a detailed overview of the project objectives, methodology, and results.

## Acknowledgments

- **Data Source**: NYPD Arrests Data (Historic 2006 - 2020).
- **Tools and Technologies**: Python, Jupyter, Tableau, scikit-learn, Pandas, Matplotlib, Seaborn.

---

Feel free to open an issue or submit a pull request if you encounter any problems or have suggestions for improvement!
## To Run
https://nyc-crime-route-prediction.streamlit.app/
