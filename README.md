
# Sales Forecasting Project

## **Objective**
The primary goal of this project is to identify products that are predicted to generate the highest revenue based on their current features and characteristics. The project involves cleaning and preprocessing raw data, building an ETL pipeline, and developing a machine learning model to estimate product performance.

---

## **Project Overview**
This project is structured to follow a data engineering and machine learning pipeline:
1. **Data Extraction**: Download and store raw sales data from Kaggle in AWS S3.
2. **Data Cleaning and Transformation**: Preprocess the data to handle missing values, invalid entries, and outliers. Engineer features relevant for sales prediction.
3. **ETL Pipeline**: Build an Extract, Transform, and Load pipeline using Python and AWS S3 to automate data flow.
4. **Machine Learning Model**: Train and evaluate a model to predict sales performance for individual products.
5. **Visualization and Reporting**: Present insights using visualizations and document findings for decision-making.

---

## **Workflow Steps**
### 1. **Define the Goal**
- Predict product sales for the upcoming year using historical sales data.
- Evaluate the model's performance using metrics like RMSE or R².

### 2. **Data Acquisition**
- Dataset Source: [Kaggle](https://www.kaggle.com/datasets/ahmedmohamed2003/retail-store-sales-dirty-for-data-cleaning)
- Example Data Fields:
  - `Transaction ID`
  - `Customer ID`
  - `Product`
  - `Category`
  - `Price Per Unit`
  - `Quantity`
  - `Transaction Date`
  - `Location`
  - `Discount Applied`

### 3. **ETL Pipeline**
- **Extract**: Download the dataset from S3 using `boto3`.
- **Transform**: Clean and preprocess the dataset using Pandas:
  - Validate fields like Transaction ID, Customer ID, and Prices.
  - Handle missing or invalid values.
  - Engineer features like seasonal trends and price-per-unit.
- **Load**:  Save the cleaned data back into S3 under the transformed/ and preprocessed/ folders.

### 4. **Machine Learning** 
- Train a predictive model using algorithms such as Linear Regression or XGBoost.
- Evaluate the model using validation techniques and refine it based on metrics.

### 5. **Visualization** --in progress
- Generate visual insights:
  - Predicted vs. actual sales performance.
  - Product rankings based on forecasted sales.

---

## **Technologies Used**
- **Programming Language**: Python
- **Cloud Platform**: AWS S3
- **Libraries**:
  - `pandas`, `numpy` for data manipulation.
  - `matplotlib`, `seaborn` for visualization. -- in progress
  - `scikit-learn`, `xgboost` for machine learning.
  - `boto3` for AWS S3 integration.
  - `dotenv` for environment variable management.

---

## **Setup and Usage**
1. Clone the repository:
   ```bash
   git clone https://github.com/username/sales-forecasting.git
   cd sales-forecasting
   ```
2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure AWS credentials in a `.env` file:
   ```
   AWS_ACCESS_KEY=your_access_key
   AWS_SECRET_KEY=your_secret_key
   ```
4. Run the ETL pipeline:
   ```bash
   python etl_pipeline.py
   ```
5. Train and evaluate the machine learning model:
   ```bash
   python train_model.py
   ```

---
## **Outputs**
1. **Cleaned Dataset**:
   - Transformed sales data stored in AWS S3 under the `transformed/` folder.
2. **Trained Model**:
   - A model trained to forecast product performance.
3. **Predictions**:
   - Forecasted sales data for the upcoming year.
   - **Top Predicted Products**:
   
     | Reconstructed_Item  | Predicted Revenue |
     |---------------------|-------------------|
     | Item_Item_25_FUR    | 24715.45          |
     | Item_Item_25_EHE    | 23073.85          |
     | Item_Item_25_BUT    | 22244.72          |
     | Item_Item_24_FUR    | 20758.97          |
     | Item_Item_25_FOOD   | 20270.84          |
     | Item_Item_22_BUT    | 19938.79          |
     | Item_Item_23_BUT    | 19286.40          |
     | Item_Item_19_MILK   | 18953.87          |
     | Item_Item_20_BUT    | 18836.56          |
     | Item_Item_23_PAT    | 18439.58          |

4. **Model Performance**:
   - **Mean Squared Error (MSE)**: 961.23
   - **R² Score**: 0.89

---

## **Future Work**
- Integrate time-series forecasting models for seasonal trends.
- Deploy the trained model using AWS Lambda for real-time predictions.
- Build a user-friendly dashboard for sales insights.

---

## **Contributors**
- **Muntaqa Maahi**  
  Data Engineer and Machine Learning Enthusiast

---

[⬆️ Back to Top](#sales-forecasting-project)