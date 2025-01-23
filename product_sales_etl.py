from dotenv import load_dotenv
import os
import boto3
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



# ----------------- Helper Functions ----------------- #
def validate_column(data, column_name, pattern, description):
    """
    Validate a column based on a regex pattern.
    """
    data[f'Valid_{column_name}'] = data[column_name].apply(lambda x: bool(re.match(pattern, str(x))) if pd.notnull(x) else False)
    invalid_rows = data[~data[f'Valid_{column_name}']]
    if not invalid_rows.empty:
        print(f"Invalid {description} detected:")
        print(invalid_rows)
    return invalid_rows

def validate_range(data, column_name, min_value=None, max_value=None, description=None):
    """
    Validate a column to ensure values fall within a specified range.
    """
    invalid_rows = data[
        (data[column_name] < min_value if min_value is not None else False) |
        (data[column_name] > max_value if max_value is not None else False)
    ]
    if not invalid_rows.empty:
        print(f"Invalid {description} detected:")
        print(invalid_rows)
    return invalid_rows

# ----------------- Load Environment and Connect to S3 ----------------- #

# Load environment variables from .env file
load_dotenv()

# Access environment variables
access_key = os.getenv("AWS_ACCESS_KEY")
secret_key = os.getenv("AWS_SECRET_KEY")

if not access_key or not secret_key:
    raise ValueError("AWS credentials are not set correctly in .env file")

print("AWS Access Key and Secret Key loaded successfully.")

# S3 details
bucket_name = "sales-forecasting-etl"
file_key = "raw/retail_store_sales.csv"

# Download file from S3
s3 = boto3.client(
    "s3",
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
)
s3.download_file(bucket_name, file_key, "local_retail_store_sales.csv")

# Load data into a Pandas DataFrame
data = pd.read_csv("local_retail_store_sales.csv")
print(data.head())
print(data.info())

# ----------------- Validate Data ----------------- #

# 1. Validate Transaction ID
transaction_pattern = r'^T[A-Z0-9]{2}_[A-Z0-9]{7}$'
validate_column(data, 'Transaction ID', transaction_pattern, 'Transaction IDs')

# 2. Validate Customer ID
customer_pattern = r'^CUST_[A-Z0-9]{2,}$'
validate_column(data, 'Customer ID', customer_pattern, 'Customer IDs')

# 3. Validate Item Format
item_pattern = r'^Item_\d+_[A-Z]+$'
validate_column(data, 'Item', item_pattern, 'Items')

# 4. Validate Price Per Unit
validate_range(data, 'Price Per Unit', min_value=0, max_value=10000, description='Prices Per Unit')

# 5. Validate Quantity
validate_range(data, 'Quantity', min_value=0, description='Quantities')

# 6. Validate Location
valid_locations = ['Online', 'In-store']
invalid_locations = data[~data['Location'].isin(valid_locations)]
if not invalid_locations.empty:
    print("Invalid Locations detected:")
    print(invalid_locations)

# 7. Validate Transaction Date
data['Transaction Date'] = pd.to_datetime(data['Transaction Date'], format='%Y-%m-%d', errors='coerce')
invalid_dates = data[data['Transaction Date'].isnull()]
if not invalid_dates.empty:
    print("Invalid Transaction Dates detected:")
    print(invalid_dates)
    data = data.dropna(subset=['Transaction Date'])

# 8. Validate Discount Applied
invalid_discounts = data[~data['Discount Applied'].isin([True, False]) & data['Discount Applied'].notnull()]
if not invalid_discounts.empty:
    print("Invalid Discount Applied values detected:")
    print(invalid_discounts)


# ----------------- Validate and Fill Data ----------------- #

# Validate Item Format
item_pattern = r'^Item_\d+_[A-Z]+$'
validate_column(data, 'Item', item_pattern, 'Items')

# Fill Missing or Invalid Items
invalid_items = data[~data['Valid_Item']]

if not invalid_items.empty:
    print("Invalid or Missing Items detected. Attempting to fill...")

    # Group by 'Category' and 'Price Per Unit' to find the most common 'Item'
    grouped_modes = (
        data.groupby(['Category', 'Price Per Unit'])['Item']
        .agg(lambda x: x.mode()[0] if not x.mode().empty else None)
        .reset_index()
        .rename(columns={'Item': 'Most_Common_Item'})
    )

    # Merge the grouped modes back into the original data
    data = data.merge(grouped_modes, on=['Category', 'Price Per Unit'], how='left')

    # Fill missing or invalid 'Item' values with the most common item for the group
    data.loc[~data['Valid_Item'], 'Item'] = data.loc[~data['Valid_Item'], 'Most_Common_Item']

    # Drop the helper column
    data.drop(columns=['Most_Common_Item'], inplace=True)

    # Check if any rows were updated --
    updated_items = data[~data['Valid_Item']]
    print("Updated Items:")
    print(updated_items[['Transaction ID', 'Category', 'Price Per Unit', 'Item']])

else:
    print("All Items are valid.")


# Infer 'Discount Applied'
expected_total = data['Price Per Unit'] * data['Quantity']
data['Discount Applied'] = data['Discount Applied'].fillna(data['Total Spent'] < expected_total)

# Infer 'Item'
grouped_modes = (
    data.groupby(['Category', 'Price Per Unit'])['Item']
    .agg(lambda x: x.mode()[0] if not x.mode().empty else None)
    .reset_index()
    .rename(columns={'Item': 'Most_Common_Item'})
)
data = data.merge(grouped_modes, on=['Category', 'Price Per Unit'], how='left')
data['Item'] = data['Item'].fillna(data['Most_Common_Item'])
data.drop(columns=['Most_Common_Item'], inplace=True)

# Infer 'Price Per Unit'
data['Price Per Unit'] = data.groupby('Category')['Price Per Unit'].transform(
    lambda x: x.fillna(x.mean())
)

# Infer 'Quantity' and 'Total Spent' using context
data['Quantity'] = data['Quantity'].fillna(data['Total Spent'] / data['Price Per Unit'])
data['Total Spent'] = data['Total Spent'].fillna(data['Price Per Unit'] * data['Quantity'])

# Infer 'Item' using grouped modes
grouped_modes = (
    data.groupby(['Category', 'Price Per Unit'])['Item']
    .agg(lambda x: x.mode()[0] if not x.mode().empty else None)
    .reset_index()
    .rename(columns={'Item': 'Most_Common_Item'})
)
data = data.merge(grouped_modes, on=['Category', 'Price Per Unit'], how='left')
data['Item'] = data['Item'].fillna(data['Most_Common_Item'])
data.drop(columns=['Most_Common_Item'], inplace=True)

# Confirm no missing values remain
print("Remaining missing values after imputation:")
print(data.isnull().sum())

# Drop all rows with any remaining NaN values
data = data.dropna()

# Confirm no missing values remain
print("Remaining missing values after dropping rows:")
print(data.isnull().sum())


# ----------------- Summary ----------------- #

print("Data validation completed.")

# Save the transformed data to a new CSV file
transformed_file_name = "transformed_retail_store_sales.csv"
data.to_csv(transformed_file_name, index=False)
print(f"Transformed data has been saved to {transformed_file_name}.")

# add to  s3 in transformed folder
transformed_file_name = "transformed_retail_store_sales.csv"
upload_key = f"transformed/{transformed_file_name}"

s3.upload_file(transformed_file_name, bucket_name, upload_key)
print(f"File uploaded to S3 bucket '{bucket_name}' under '{upload_key}'")


#---------- generate report for missing data --------
def generate_missing_values_report(data):
    """
    Generate a report of missing (NaN) values in the dataset.
    """
    total = data.isnull().sum()
    percent = (data.isnull().sum() / len(data)) * 100
    missing_report = pd.DataFrame({
        'Total Missing': total,
        'Percent Missing (%)': percent
    })
    return missing_report[missing_report['Total Missing'] > 0].sort_values(by='Total Missing', ascending=False)

# Generate and display the report
missing_report = generate_missing_values_report(data)
print("Missing Values Report:")
print(missing_report)



# time to create the machine learning model


# ------------ Pre processing -----------------
# One hot encoding
data = pd.read_csv("transformed_retail_store_sales.csv")
# print(data.head())
data = pd.get_dummies(data, columns=['Category', 'Item', 'Payment Method', 'Location'], drop_first=True)

# #breaking up data for more features
# print("----------checking date type -------------")
# print(data['Transaction Date'].dtype)
# print("----------------------------------------------")
# data['Year'] = data['Transaction Date'].dt.year
# data['Month'] = data['Transaction Date'].dt.month


#scale and normalize so that data so model does not assume the wrong weight to values
scaler = StandardScaler()
data[['Price Per Unit', 'Quantity']] = scaler.fit_transform(data[['Price Per Unit', 'Quantity']])


preprocessed_file_name = "preprocessed_retail_store_sales.csv"
data.to_csv(preprocessed_file_name, index=False)
print(f"Preprocessed data has been saved to {preprocessed_file_name}.")


#split the data 
# Define X (features) and y (target)
X = data.drop(columns=['Total Spent', 'Transaction ID', 'Customer ID', 'Transaction Date'])
y = data['Total Spent']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#-------------- Train data ------------

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)


# ------------- Predict ------------
# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")



target_variance = y.var()
print("----------------------------------")
print(f"Target Variance: {target_variance}")




# Predict future sales for each row
X_test = data.drop(columns=['Total Spent', 'Transaction ID', 'Customer ID', 'Transaction Date'])
predicted_sales = model.predict(X_test)

# Add predictions to the dataset
data['Predicted Total Spent'] = predicted_sales

# Reconstruct the original 'Item' column from one-hot encoded columns
item_columns = [col for col in data.columns if col.startswith('Item_')]
data['Reconstructed_Item'] = data[item_columns].idxmax(axis=1)

# Aggregate predictions by reconstructed 'Item'
future_performance = data.groupby('Reconstructed_Item').agg({
    'Predicted Total Spent': 'sum'
}).reset_index()

# Sort by predicted total revenue
# Display the top-performing items with "Predicted Revenue" in the output
future_performance.rename(columns={'Predicted Total Spent': 'Predicted Revenue'}, inplace=True)

print("Top predicted products for the upcoming year:")
print(future_performance.head(10))

