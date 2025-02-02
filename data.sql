-- 1. First, import cleaned_ecommerce_customer_data_with_metrics.csv as a table in the database. 
-- Drop the final table if it already exists to avoid duplicates.
DROP TABLE IF EXISTS ecommerce_metrics;

-- 2. Create the new table for the cleaned data with an additional age_group column.
CREATE TABLE ecommerce_metrics (
    customer_id INTEGER,
    purchase_date TEXT,
    product_category TEXT,
    product_price REAL,
    quantity INTEGER,
    total_purchase_amount REAL,
    payment_method TEXT,
    customer_age INTEGER,
    returns REAL,
    gender TEXT,
    churn INTEGER,
    customer_lifetime_value REAL,
    age_group TEXT
);

-- 3. Insert data from the original imported table into the new ecommerce_metrics table.
INSERT INTO ecommerce_metrics
SELECT 
    "Customer ID",
    "Purchase Date",
    "Product Category",
    "Product Price",
    Quantity,
    "Total Purchase Amount",
    "Payment Method",
    "Customer Age",
    Returns,
    Gender,
    Churn,
    "Customer Lifetime Value (CLV)",
    CASE
        WHEN "Customer Age" < 25 THEN 'Under 25'
        WHEN "Customer Age" BETWEEN 25 AND 34 THEN '25-34'
        WHEN "Customer Age" BETWEEN 35 AND 44 THEN '35-44'
        WHEN "Customer Age" BETWEEN 45 AND 54 THEN '45-54'
        WHEN "Customer Age" >= 55 THEN '55+'
        ELSE 'Unknown'
    END
FROM cleaned_ecommerce_customer_data_with_metrics;

-- 4. Create an index on customer_id for faster queries.
CREATE INDEX idx_customer_id ON ecommerce_metrics (customer_id);

-- 5. Drop the original table to clean up the database.
DROP TABLE IF EXISTS cleaned_ecommerce_customer_data_with_metrics;

-- 6. Create a new table for storing the predictions from predictions.csv.
DROP TABLE IF EXISTS ecommerce_predictions;
CREATE TABLE ecommerce_predictions (
    customer_id INTEGER,
    actual_churn INTEGER,
    predicted_churn INTEGER,
    actual_CLV REAL,
    predicted_CLV REAL
);

-- 7. Insert data from the predictions CSV file into the predictions table.
INSERT INTO ecommerce_predictions
SELECT
    "Customer ID",
    "Actual Churn",
    "Predicted Churn",
    "Actual CLV",
    "Predicted CLV"
FROM predictions;

DROP TABLE IF EXISTS predictions;
