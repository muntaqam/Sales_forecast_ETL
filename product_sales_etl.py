from dotenv import load_dotenv
import os
import boto3
import pandas as pd
import re

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
