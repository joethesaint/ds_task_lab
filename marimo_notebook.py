# /// script
# dependencies = ["pinecone", "google-generativeai"]
# ///

import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _(re):
    def redact_pii(text):
        """Simple PII redaction helper.

            Replaces common email patterns, long numeric sequences, and phone-like numbers with placeholders.
            This is a light-weight safeguard; do not rely on it for legal-grade PII removal.
        """
        if text is None:
            return text
        # redact emails
        text = re.sub(r"\b[\w.%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[REDACTED_EMAIL]", text)
        # redact long numeric sequences (e.g., IDs)
        text = re.sub(r"\b\d{9,}\b", "[REDACTED_NUMBER]", text)
        # redact phone-like patterns
        text = re.sub(r"\+?\d[\d\-\s]{7,}\d", "[REDACTED_PHONE]", text)
        return text
    return


@app.cell
def _():
    import subprocess
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <a href="https://colab.research.google.com/github/joethesaint/ds_task_lab/blob/main/notebook.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Task
    Summarize the `README.md` file located at `/content/ds_task_1ab/README.md` and propose next steps for the project.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Read README

    ### Subtask:
    Read the content of the `README.md` file located at `/content/ds_task_1ab/README.md`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reasoning**:
    To read the content of the `README.md` file, I will use Python's built-in file handling to open the file in read mode and print its content.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Summary of `README.md`

    The project aims to develop a comprehensive solution for product recommendation, OCR-based query processing, and image-based product detection, divided into four main modules:

    **Module 1: Data Preparation and Backend Setup**
    - **Task 1: E-commerce Dataset Cleaning**: Clean the dataset (remove duplicates, handle missing values, standardize formats).
    - **Task 2: Vector Database Creation**: Set up a Pinecone vector database.
    - **Task 3: Similarity Metrics Selection**: Choose and justify similarity metrics.
    - **Endpoint 1: Product Recommendation Service**: Handle natural language queries for product recommendations.

    **Module 2: OCR and Web Scraping**
    - **Task 4: OCR Functionality Implementation**: Develop OCR to extract text from images.
    - **Task 5: Web Scraping for Product Images**: Scrape product images for `CNN_Model_Train_Data.csv`.
    - **Endpoint 2: OCR-Based Query Processing**: Process handwritten queries from images.

    **Module 3: CNN Model Development**
    - **Task 6: CNN Model Training**: Develop and train a CNN model from scratch using `CNN_Model_Train_Data.csv` to identify products from images.
    - **Endpoint 3: Image-Based Product Detection**: Use the CNN model to identify products and match them using the vector database.

    **Module 4: Frontend Development and Integration**
    - **Frontend Page 1: Text Query Interface**
    - **Frontend Page 2: Image Query Interface**
    - **Frontend Page 3: Product Image Upload Interface**

    **Instructions for Presentation** include incremental report writing and video documentation for each module.

    **Instructions for Coding** emphasize class-based implementation, best practices (ACID, Modularity), packaging, and directory structure. The **Tech Stack** specifies Flask for the backend and Pinecone for the vector database.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Proposed Next Steps for the Project

    Given the comprehensive overview in the `README.md`, the project should proceed in a modular fashion, following the outlined structure. The immediate next steps should focus on completing **Module 1: Data Preparation and Backend Setup**.

    1.  **Start Module 1: Data Preparation and Backend Setup**
        *   **Task 1: E-commerce Dataset Cleaning**: The first priority is to clean the existing e-commerce dataset. This includes:
            *   Identifying and removing duplicate entries.
            *   Handling missing values appropriately (e.g., imputation, removal).
            *   Standardizing data formats (e.g., product names, descriptions, categories) to ensure consistency.
            *   *Action*: Load the `dataset.csv` and `CNN_Model_Train_Data.csv` into DataFrames and perform initial data quality checks.
        *   **Task 2: Vector Database Creation (Pinecone)**: Once the dataset is clean, the next step is to prepare for vectorization and set up the vector database.
            *   *Action*: Define the schema for product vectors and establish connection with Pinecone.
        *   **Task 3: Similarity Metrics Selection**: Concurrently with vector database setup, research and select the most appropriate similarity metrics for product comparison.
            *   *Action*: Evaluate cosine similarity, dot product, and other relevant metrics based on the nature of the product data.
        *   **Endpoint 1: Product Recommendation Service**: Begin planning the architecture and implementation details for the natural language query endpoint.
            *   *Action*: Design the API interface and consider safeguards against bad queries and sensitive data exposure.

    2.  **Plan for Module 2: OCR and Web Scraping**
        *   While Module 1 is in progress, begin preliminary research for OCR tools (e.g., Tesseract) and web scraping strategies. Identify potential data sources for product images for `CNN_Model_Train_Data.csv`.

    3.  **Establish Version Control and Reporting Protocols**
        *   Ensure all team members are familiar with Git for version control. Adhere to the specified incremental report writing and video documentation requirements for each module's completion.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary:

    ### Q&A
    The task asked to summarize the `README.md` file and propose next steps for the project. The summary of the `README.md` has been provided, outlining the project's structure, modules, and technical requirements. Additionally, a detailed proposal for the immediate next steps has been given, prioritizing Module 1 tasks.

    ### Data Analysis Key Findings
    *   The project is structured into four main modules:
        *   **Module 1: Data Preparation and Backend Setup**: Encompasses e-commerce dataset cleaning, Pinecone vector database creation, similarity metrics selection, and a product recommendation service.
        *   **Module 2: OCR and Web Scraping**: Focuses on implementing OCR functionality and web scraping product images.
        *   **Module 3: CNN Model Development**: Involves training a CNN model from scratch for image-based product detection.
        *   **Module 4: Frontend Development and Integration**: Includes creating three distinct frontend pages for text queries, image queries, and product image uploads.
    *   The `README.md` specifies instructions for presentation (incremental report writing, video documentation for each module) and coding best practices (class-based implementation, ACID principles, modularity, packaging, and directory structure).
    *   The specified tech stack includes Flask for the backend and Pinecone for the vector database.

    ### Insights or Next Steps
    *   The project should immediately commence with **Module 1: Data Preparation and Backend Setup**, prioritizing e-commerce dataset cleaning (handling duplicates, missing values, standardization), setting up the Pinecone vector database, selecting appropriate similarity metrics, and designing the Product Recommendation Service endpoint.
    *   Concurrently, initiate preliminary research for **Module 2: OCR and Web Scraping** tools and strategies, and establish robust version control and reporting protocols as outlined in the `README.md` to ensure project cohesion and documentation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Task
    Load 'dataset.csv' and 'CNN_Model_Train_Data.csv' into pandas DataFrames to begin initial inspection and data quality checks.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load Datasets

    ### Subtask:
    Load 'dataset.csv' and 'CNN_Model_Train_Data.csv' into pandas DataFrames to begin initial inspection and data quality checks.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reasoning**:
    To load the datasets and perform initial inspection, I will use pandas to read the CSV files into DataFrames and then display the first few rows and column information for each.
    """)
    return


@app.cell
def _():
    import pandas as pd

    # Load 'dataset.csv' into df_dataset
    df_dataset = pd.read_csv('data/dataset.csv')

    # Load 'CNN_Model_Train_Data.csv' into df_cnn_train_data
    df_cnn_train_data = pd.read_csv('data/CNN_Model_Train_Data.csv')

    print("\n--- df_dataset Head ---\n")
    print(df_dataset.head())
    print("\n--- df_dataset Info ---\n")
    df_dataset.info()

    print("\n--- df_cnn_train_data Head ---\n")
    print(df_cnn_train_data.head())
    print("\n--- df_cnn_train_data Info ---\n")
    df_cnn_train_data.info()
    return df_cnn_train_data, df_dataset, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Initial Data Inspection Summary

    **df_dataset:**
    -   **Rows and Columns**: The `df_dataset` DataFrame contains 541,909 entries and 8 columns.
    -   **Data Types**: All columns are currently of `object` dtype, which suggests that some columns that should be numerical (e.g., `Quantity`, `UnitPrice`, `CustomerID`) or datetime (`InvoiceDate`) need type conversion.
    -   **Missing Values**:
        -   `Description`: 540,884 non-null values out of 541,909, indicating ~1,025 missing values.
        -   `CustomerID`: 433,909 non-null values out of 541,909, indicating ~108,000 missing values.
    -   **Data Anomalies/Special Characters**:
        -   `StockCode`: Contains special characters like `ö` and `^` (e.g., `ö84406B^`).
        -   `Quantity`: Contains special characters like `@` (e.g., `6@`).
        -   `CustomerID`: Contains special characters like `&` and `#` (e.g., `&17850.0#`).
        -   `Country`: Contains special characters like `X`, `x`, `Y`, `y`, and emojis (e.g., `XxYUnited Kingdom☺️`).
        -   `InvoiceNo`: Contains numerical values and also appears to have 'object' type which might indicate non-numeric entries, possibly for cancelled orders (not explicitly seen in head, but common in such datasets).
        -   `UnitPrice`: Appears to be a float but is of 'object' type, suggesting possible non-numeric entries or formatting issues.

    **df_cnn_train_data:**
    -   **Rows and Columns**: The `df_cnn_train_data` DataFrame is small, with 10 entries and 1 column.
    -   **Data Types**: The `StockCode` column is of `object` dtype.
    -   **Missing Values**: No missing values.
    -   **Data Anomalies/Special Characters**:
        -   `StockCode`: Contains special characters like `ö` and `^` (e.g., `ö22384^`).

    **Next Steps based on Initial Inspection:**
    Based on these observations, significant data cleaning will be required for `df_dataset` to address incorrect data types, missing values, and special characters. `df_cnn_train_data` also requires cleaning of special characters in `StockCode`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Initial Data Inspection Summary

    **df_dataset:**
    -   **Rows and Columns**: The `df_dataset` DataFrame contains 541,909 entries and 8 columns.
    -   **Data Types**: All columns are currently of `object` dtype, which suggests that some columns that should be numerical (e.g., `Quantity`, `UnitPrice`, `CustomerID`) or datetime (`InvoiceDate`) need type conversion.
    -   **Missing Values**:
        -   `Description`: 540,884 non-null values out of 541,909, indicating ~1,025 missing values.
        -   `CustomerID`: 433,909 non-null values out of 541,909, indicating ~108,000 missing values.
    -   **Data Anomalies/Special Characters**:
        -   `StockCode`: Contains special characters like `ö` and `^` (e.g., `ö84406B^`).
        -   `Quantity`: Contains special characters like `@` (e.g., `6@`).
        -   `CustomerID`: Contains special characters like `&` and `#` (e.g., `&17850.0#`).
        -   `Country`: Contains special characters like `X`, `x`, `Y`, `y`, and emojis (e.g., `XxYUnited Kingdom☺️`).
        -   `InvoiceNo`: Contains numerical values and also appears to have 'object' type which might indicate non-numeric entries, possibly for cancelled orders (not explicitly seen in head, but common in such datasets).
        -   `UnitPrice`: Appears to be a float but is of 'object' type, suggesting possible non-numeric entries or formatting issues.

    **df_cnn_train_data:**
    -   **Rows and Columns**: The `df_cnn_train_data` DataFrame is small, with 10 entries and 1 column.
    -   **Data Types**: The `StockCode` column is of `object` dtype.
    -   **Missing Values**: No missing values.
    -   **Data Anomalies/Special Characters**:
        -   `StockCode`: Contains special characters like `ö` and `^` (e.g., `ö22384^`).

    **Next Steps based on Initial Inspection:**
    Based on these observations, significant data cleaning will be required for `df_dataset` to address incorrect data types, missing values, and special characters. `df_cnn_train_data` also requires cleaning of special characters in `StockCode`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Clean E-commerce Dataset

    ### Subtask:
    Perform data cleaning on 'dataset.csv' (df_dataset), including identifying and removing duplicate entries, handling missing values, and standardizing data formats (e.g., product names, descriptions, categories, numerical columns, and date columns).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reasoning**:
    To begin the data cleaning process, I will first remove leading/trailing whitespace from all string columns in `df_dataset` as specified in the instructions. This is a common initial step for data standardization.
    """)
    return


@app.cell
def _(df_dataset):
    for col in df_dataset.select_dtypes(include='object').columns:
        df_dataset[col] = df_dataset[col].astype(str).str.strip()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reasoning**:
    Following the data cleaning instructions, I will clean the 'Quantity' column by removing non-numeric characters and then converting it to a numeric data type, coercing errors to NaN.
    """)
    return


@app.cell
def _(df_dataset, pd):
    df_dataset['Quantity'] = df_dataset['Quantity'].astype(str).str.replace(r'[^\d.-]', '', regex=True)
    df_dataset['Quantity'] = pd.to_numeric(df_dataset['Quantity'], errors='coerce')

    print("--- df_dataset Quantity Info after cleaning ---")
    df_dataset['Quantity'].info()
    print("--- df_dataset Quantity head after cleaning ---")
    print(df_dataset['Quantity'].head())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reasoning**:
    Following the data cleaning instructions, I will clean the 'UnitPrice' column by converting it to a numeric data type, coercing errors to NaN.
    """)
    return


@app.cell
def _(df_dataset, pd):
    df_dataset['UnitPrice'] = pd.to_numeric(df_dataset['UnitPrice'], errors='coerce')

    print("---> df_dataset UnitPrice Info after cleaning ---")
    df_dataset['UnitPrice'].info()
    print("---> df_dataset UnitPrice head after cleaning ---")
    print(df_dataset['UnitPrice'].head())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reasoning**:
    Following the data cleaning instructions, I will clean the 'CustomerID' column by removing non-numeric characters and converting it to a numeric data type, coercing errors to NaN.
    """)
    return


@app.cell
def _(df_dataset, pd):
    df_dataset['CustomerID'] = df_dataset['CustomerID'].astype(str).str.replace(r'[^\d.]', '', regex=True)
    df_dataset['CustomerID'] = pd.to_numeric(df_dataset['CustomerID'], errors='coerce')

    print("--- df_dataset CustomerID Info after cleaning ---")
    df_dataset['CustomerID'].info()
    print("--- df_dataset CustomerID head after cleaning ---")
    print(df_dataset['CustomerID'].head())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reasoning**:
    Following the data cleaning instructions, I will convert the 'InvoiceDate' column to a datetime data type to enable time-based analysis.
    """)
    return


@app.cell
def _(df_dataset, pd):
    df_dataset['InvoiceDate'] = pd.to_datetime(df_dataset['InvoiceDate'], errors='coerce')

    print("--- df_dataset InvoiceDate Info after cleaning ---")
    df_dataset['InvoiceDate'].info()
    print("--- df_dataset InvoiceDate head after cleaning ---")
    print(df_dataset['InvoiceDate'].head())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reasoning**:
    Following the data cleaning instructions, I will clean the 'StockCode' column by removing special characters using regular expressions.
    """)
    return


@app.cell
def _(df_cnn_train_data, df_dataset):
    df_dataset['StockCode'] = df_dataset['StockCode'].astype(str).str.replace(r'[^a-zA-Z0-9]', '', regex=True)
    df_cnn_train_data['StockCode'] = df_cnn_train_data['StockCode'].astype(str).str.replace(r'[^a-zA-Z0-9]', '', regex=True)

    print("--- df_dataset StockCode head after cleaning ---")
    print(df_dataset['StockCode'].head())
    print("\n--- df_cnn_train_data StockCode head after cleaning ---")
    print(df_cnn_train_data['StockCode'].head())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reasoning**:
    Following the data cleaning instructions, I will clean the 'Country' column by removing non-alphabetic characters and emojis, and then standardize the country names. I will first use a regular expression to remove characters that are not letters or spaces, then use `replace` to standardize common variations of 'United Kingdom'.
    """)
    return


@app.cell
def _(df_dataset):
    df_dataset['Country'] = df_dataset['Country'].astype(str).str.replace(r'[^a-zA-Z\s]', '', regex=True)
    df_dataset['Country'] = df_dataset['Country'].str.replace('United Kingdom', 'United Kingdom', regex=False).str.strip()

    print("--- df_dataset Country head after cleaning ---")
    print(df_dataset['Country'].head())
    print("\n--- Unique countries after cleaning ---")
    print(df_dataset['Country'].unique())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reasoning**:
    The previous attempt to clean the 'Country' column did not fully remove the 'XxY' prefixes because 'X', 'x', 'Y', 'y' are considered alphabetic characters. To accurately follow the instruction to remove these specific characters along with other non-alphabetic characters and emojis, I will apply a multi-step cleaning process: first, explicitly remove `X`, `x`, `Y`, `y` characters, then remove any other non-alphabetic characters and emojis, and finally, standardize spacing and apply title casing for consistency.
    """)
    return


@app.cell
def _(df_dataset):
    df_dataset['Country'] = df_dataset['Country'].astype(str).str.replace(r'[XxYy]', '', regex=True)

    df_dataset['Country'] = df_dataset['Country'].str.replace(r'[^a-zA-Z\s]', '', regex=True)

    df_dataset['Country'] = df_dataset['Country'].str.strip()
    df_dataset['Country'] = df_dataset['Country'].str.replace(r'\s+', ' ', regex=True)

    df_dataset['Country'] = df_dataset['Country'].str.title()

    print("--- df_dataset Country head after cleaning ---")
    print(df_dataset['Country'].head())
    print("\n--- Unique countries after cleaning ---")
    print(df_dataset['Country'].unique())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reasoning**:
    Following the data cleaning instructions, I will address missing values by filling missing 'Description' values with 'Unknown Product' as a placeholder.
    """)
    return


@app.cell
def _(df_dataset):
    df_dataset['Description'].fillna('Unknown Product', inplace=True)

    print("--- df_dataset Description Info after handling missing values ---")
    df_dataset['Description'].info()
    print("--- df_dataset Description head after handling missing values ---")
    print(df_dataset['Description'].head())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reasoning**:
    The previous code produced a FutureWarning related to `inplace=True` when modifying a DataFrame slice. To address this and ensure future compatibility, I will re-write the `fillna` operation by directly assigning the result back to the 'Description' column, which is the recommended practice.
    """)
    return


@app.cell
def _(df_dataset):
    df_dataset['Description'] = df_dataset['Description'].fillna('Unknown Product')

    print("--- df_dataset Description Info after handling missing values ---")
    df_dataset['Description'].info()
    print("--- df_dataset Description head after handling missing values ---")
    print(df_dataset['Description'].head())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reasoning**:
    Following the data cleaning instructions, I will address missing values in the 'CustomerID' column. Given that `CustomerID` is essential for personalized recommendations, and imputing it would create artificial users, the most appropriate strategy is to remove rows where `CustomerID` is missing.
    """)
    return


@app.cell
def _(df_dataset):
    df_dataset.dropna(subset=['CustomerID'], inplace=True)

    print("--- df_dataset CustomerID Info after handling missing values ---")
    df_dataset['CustomerID'].info()
    print("--- df_dataset head after handling missing CustomerID ---")
    print(df_dataset.head())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reasoning**:
    Following the data cleaning instructions, the final step for `df_dataset` is to identify and remove any duplicate rows across all columns to ensure data integrity and uniqueness.
    """)
    return


@app.cell
def _(df_dataset):
    initial_rows = df_dataset.shape[0]
    df_dataset.drop_duplicates(inplace=True)
    final_rows = df_dataset.shape[0]
    duplicates_removed = initial_rows - final_rows

    print(f"--- Duplicate Rows Removed: {duplicates_removed} ---")
    print("--- df_dataset Info after removing duplicates ---")
    df_dataset.info()
    print("--- df_dataset head after removing duplicates ---")
    print(df_dataset.head())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Task
    Initialize Pinecone by installing the `pinecone-client` library and then using your actual Pinecone API key and environment to configure the Pinecone client.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Initialize Pinecone

    ### Subtask:
    Install the `pinecone-client` library and initialize the Pinecone client with your API key and environment.
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: pinecone !pip install pinecone
    print("pinecone library installation initiated.")
    return


app._unparsable_cell(
    r"""
    from pinecone import Pinecone

    # Initialize Pinecone using environment variables (safer than hardcoding keys)
    import os

    pinecone_api_key = os.environ.get('PINECONE_API_KEY')
    pinecone_environment = os.environ.get('PINECONE_ENV', 'gcp-starter')

    if not pinecone_api_key:
        # Follow rules: do not hardcode secrets; instruct the user to set env vars instead.
        print(\"PINECONE_API_KEY not set. Skipping Pinecone initialization. Set PINECONE_API_KEY in the environment to enable Pinecone.\")
        return (None,)

    pinecone = Pinecone(api_key=pinecone_api_key, environment=pinecone_environment)

    print(\"Pinecone client initialized (api key sourced from env).\")
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Create Pinecone Index

    ### Subtask:
    Create a new Pinecone index or connect to an existing one, defining the index name, dimension, and metric type suitable for product vector similarity search.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Pinecone Index Parameters

    To create or connect to a Pinecone index, we need to define the following parameters:

    1.  **Index Name**: `ecommerce-product-vectors` will be used as a descriptive name for our product vector index.
    2.  **Dimension**: While we haven't selected an embedding model yet, a common dimension for many general-purpose embedding models (like those from OpenAI, Sentence Transformers, etc.) is 1536 or 768. For now, I will use **1536** as a placeholder, which can be adjusted later once a specific embedding model is chosen. If we were to use a model like `text-embedding-ada-002`, the dimension would be 1536.
    3.  **Metric Type**: For product recommendation, **cosine similarity** is a widely used and effective metric. It measures the cosine of the angle between two vectors, indicating how similar their orientations are. `euclidean` (Euclidean distance) or `dotproduct` could also be options, but `cosine` often performs well for semantic similarity tasks.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reasoning**:
    Now that the Pinecone client is initialized and the index parameters are defined, I will write the code to check for the existence of the specified Pinecone index and either connect to it or create a new one, as per the instructions.
    """)
    return


app._unparsable_cell(
    r"""
    from pinecone import ServerlessSpec
    import time
    import os

    # Use environment variable for the Pinecone key. If it's not set, abort this step safely.
    pinecone_api_key_1 = os.environ.get('PINECONE_API_KEY')
    if not pinecone_api_key_1:
        print(\"PINECONE_API_KEY not found in environment. Skipping index creation/connection. Set PINECONE_API_KEY to proceed.\")
        return (time,)

    # ⚠️ SECURITY: API key is read from environment; never hardcode in source.
    pc = Pinecone(api_key=pinecone_api_key_1)
    index_name = 'ecommerce-product-vectors'
    # 1. Initialize Pinecone
    dimension = 1536
    metric_type = 'cosine'
    existing_indexes = [index_info['name'] for index_info in pc.list_indexes()]
    if index_name not in existing_indexes:
        print(f\"Creating new index '{index_name}'...\")
        pc.create_index(name=index_name, dimension=dimension, metric=metric_type, spec=ServerlessSpec(cloud='aws', region='us-east-1'))
    # 2. Correctly check existing indexes
    # list_indexes() returns an object, so we must extract the names
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        print(f\"Index '{index_name}' created successfully.\")
    else:
        print(f\"Index '{index_name}' already exists. Connecting to it...\")
    index = pc.Index(index_name)
    print(f\"Successfully connected to index '{index_name}'.\")
    print('Index stats:')
    # 3. Connect to the index
    print(index.describe_index_stats())  # 'us-east-1' is the most standard region for AWS Free Tier Serverless  # Wait a moment for the index to initialize
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Define Product Vector Schema

    ### Subtask:
    Define the schema and metadata structure for the product vectors that will be stored in the Pinecone index, ensuring it aligns with the e-commerce dataset's features.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Product Vector Metadata Schema for Pinecone

    Based on the `df_dataset` and the requirements for product recommendation and search, the following metadata fields will be included alongside each product's vector embedding in the Pinecone index:

    1.  **`StockCode`**:
        *   **Rationale**: This is a unique identifier for each product. It is crucial for retrieving specific product details from the main `df_dataset` once a vector search returns relevant product vectors. It allows for direct lookup and linking back to the original product information.
        *   **Suitability**: The `StockCode` has already been cleaned to remove special characters and is suitable for direct storage as a string. It will serve as a primary key for product identification.

    2.  **`Description`**:
        *   **Rationale**: The product description provides rich textual information about the product. While the vector itself will capture semantic meaning from the description, storing the original description allows for human-readable display in recommendation results and can be used for keyword-based filtering or display in the frontend.
        *   **Suitability**: The `Description` column has been handled for missing values (filled with 'Unknown Product') and is suitable for direct storage as a string. Further text cleaning (e.g., lowercasing, removing extra spaces) might be considered if exact string matching is needed for filtering, but for display, the current state is sufficient.

    3.  **`Country`**:
        *   **Rationale**: The `Country` field indicates the origin or target market for a product, which can be valuable for filtering recommendations by region or for understanding geographical sales patterns. For example, a user might want to filter products available only in their country.
        *   **Suitability**: The `Country` column has been cleaned and standardized to remove special characters and normalize names (e.g., 'United Kingdom'). It is suitable for storage as a string and can be directly used for filtering or grouping product recommendations.

    4.  **`UnitPrice`**:
        *   **Rationale**: The unit price is a critical attribute for product comparison and filtering (e.g., filtering products within a certain price range). Including it in metadata allows for dynamic price-based filtering post-vector search without needing to join with the original dataset.
        *   **Suitability**: The `UnitPrice` has been converted to a numeric type (float64) and is suitable for direct storage. Missing values in `UnitPrice` were coerced to `NaN` during cleaning; these can be handled during indexing (e.g., excluded or given a default value) if products without a price should still be searchable or filterable.

    5.  **`Quantity`**:
        *   **Rationale**: The quantity of items in an invoice line can be an indicator of popularity or bulk purchase behavior. While not directly for filtering products in a recommendation system, it might be useful for aggregation or as an additional display metric.
        *   **Suitability**: The `Quantity` has been cleaned and converted to an integer type. It is suitable for direct storage.


    These fields are selected to provide a balance between essential identification (`StockCode`), descriptive content (`Description`), contextual information (`Country`), and quantifiable attributes (`UnitPrice`, `Quantity`), enabling flexible search, filtering, and display capabilities for the product recommendation service.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prepare Database for Embeddings

    ### Subtask:
    Prepare the database for storing product embeddings, which involves defining how product information will be converted into vector representations suitable for Pinecone.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1. Identify Columns for Embeddings

    For generating product embeddings, the following columns from `df_dataset` are relevant:

    *   **`Description`**: This is the most crucial column as it contains detailed textual information about the product. Semantic similarity will largely depend on the quality of this text.
    *   **`StockCode`**: While primarily an identifier, if used in combination with the description, it might provide a unique context. However, given its format after cleaning (alphanumeric only), its direct semantic contribution might be limited unless mapped to specific product categories or types.
    *   **`Country`**: This column indicates the country of origin or sale. While not directly describing the product, it could be a valuable piece of metadata to filter or refine recommendations based on regional preferences. For embeddings, it might be less directly useful than `Description`.

    Therefore, `Description` will be the primary source for generating semantic embeddings, and `StockCode` and `Country` can serve as valuable metadata fields for filtering and enriching search results.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2. Further Text Cleaning and Preprocessing

    Based on the current state of the `df_dataset` after initial cleaning, some further text cleaning and preprocessing steps are beneficial before generating embeddings:

    *   **`Description`**: While `str.strip()` was applied, the `Description` column still contains some special characters (e.g., '$', '.') and inconsistent casing. For optimal embedding quality, these should be addressed:
        *   **Lowercasing**: Convert all text to lowercase to treat words like "Holder" and "holder" as the same.
        *   **Punctuation Removal**: Remove remaining punctuation marks that might not contribute to semantic meaning.
        *   **Special Character Removal**: Ensure any lingering non-alphanumeric characters (beyond standard text punctuation) are removed.
        *   **Redundant Spaces**: Consolidate multiple spaces into a single space.

    *   **`StockCode`**: This column has been cleaned to be purely alphanumeric. For embedding purposes, it's generally best used as a unique identifier or metadata. Combining it directly with `Description` for embedding might introduce noise unless there's a clear semantic link in the `StockCode` itself. It will be kept as is for now and considered as metadata.

    *   **`Country`**: This column has been cleaned to be in title case with special characters removed. It is suitable for use as metadata directly. For embedding, it would likely be handled separately (e.g., one-hot encoded or embedded individually) rather than combined directly with product descriptions, as its semantic context is different.

    **Conclusion for Preprocessing**: The primary focus for further cleaning will be on the `Description` column to ensure it is as clean and normalized as possible for robust embedding generation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reasoning**:
    To ensure the `Description` column is clean and ready for embedding generation, I will apply the identified preprocessing steps: lowercasing, removing special characters and punctuation, and consolidating spaces.
    """)
    return


@app.cell
def _(df_dataset):
    import re

    # Lowercasing, removing special characters/punctuation, and consolidating spaces
    df_dataset['Description'] = df_dataset['Description'].astype(str).str.lower()
    df_dataset['Description'] = df_dataset['Description'].apply(lambda x: re.sub(r'[^a-z0-9\s]', '', x))
    df_dataset['Description'] = df_dataset['Description'].apply(lambda x: re.sub(r'\s+', ' ', x)).str.strip()

    print("--- df_dataset Description head after comprehensive cleaning ---")
    print(df_dataset['Description'].head())
    return (re,)


@app.cell
def _(df_dataset):
    print("--- Verified df_dataset Description column (head) ---")
    print(df_dataset['Description'].head())

    print("\n--- Verified df_dataset Info (after all cleaning) ---")
    df_dataset.info()

    print("\n--- Verified df_dataset Head (after all cleaning) ---")
    print(df_dataset.head())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Select Embedding Model

    ### Subtask:
    Identify and select an appropriate pre-trained embedding model (e.g., a Sentence Transformer model) suitable for generating semantic vectors from product descriptions, considering factors like embedding dimension and performance on e-commerce text.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Selected Embedding Model: `text-embedding-ada-002` (OpenAI)

    **Justification for Selection:**

    1.  **Embedding Dimension Match (1536)**: Our Pinecone index has been configured with a dimension of 1536. The `text-embedding-ada-002` model from OpenAI natively produces embeddings of this exact dimension, ensuring direct compatibility without any need for dimensionality reduction, padding, or re-configuring the Pinecone index.

    2.  **High Performance for Semantic Similarity**: This model is a state-of-the-art embedding model known for its strong performance in various semantic search and text similarity tasks. It excels at capturing the nuanced semantic meaning of text, which is crucial for accurate product recommendations based on descriptions.

    3.  **Suitability for E-commerce Text**: `text-embedding-ada-002` has been trained on a vast corpus of text, making it robust for understanding diverse vocabulary and concepts present in e-commerce product descriptions, even with slight variations or informal language. It effectively captures relationships between different products.

    4.  **Industry Standard and Ecosystem**: OpenAI embeddings are widely adopted in industry, benefiting from continuous improvements and a robust support ecosystem. This reduces potential issues during integration and scaling.

    **Required Library and Model Name:**

    *   **Library**: `openai` (Python client library for OpenAI API)
    *   **Model Name for API calls**: `text-embedding-ada-002`

    *Note: Using this model will require an OpenAI API key for authentication.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate Embeddings

    ### Subtask:
    Generate embeddings for each cleaned product description in `df_dataset` using the selected `text-embedding-ada-002` model, preparing the data for storage in the Pinecone index.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reasoning**:
    To generate embeddings for the product descriptions, I need to import the OpenAI client, set the API key, define an embedding function, and then apply it to the 'Description' column of `df_dataset`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Task
    ## Select Embedding Model (Gemini)

    ### Subtask:
    Identify and select an appropriate Gemini embedding model suitable for generating semantic vectors from product descriptions, considering factors like embedding dimension and performance on e-commerce text.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Select Embedding Model (Gemini)

    ### Subtask:
    Identify and select an appropriate Gemini embedding model suitable for generating semantic vectors from product descriptions, considering factors like embedding dimension and performance on e-commerce text.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Selected Embedding Model: `text-embedding-004` (Gemini)

    **Justification for Selection:**

    1.  **High Performance for Semantic Similarity**: The `text-embedding-004` model is a highly capable embedding model from Google's Gemini family, designed to produce high-quality semantic representations of text. It is well-suited for capturing the nuanced meaning in product descriptions, which is essential for accurate similarity searches and recommendations in an e-commerce context.

    2.  **Suitability for E-commerce Text**: Gemini models, especially `text-embedding-004`, have been trained on vast and diverse datasets, making them robust in understanding various types of text, including detailed product descriptions, features, and informal language often found in e-commerce. This ensures that the embeddings will effectively represent product attributes and enable discovery of semantically related items.

    3.  **Embedding Dimension**: `text-embedding-004` typically produces embeddings with a dimension of **768**. While our Pinecone index was initially set up with 1536, it's important to note that the dimension for Pinecone can be adjusted or a new index created to match the model's output. If we strictly need to maintain 1536, another model or method would be required. However, 768 is a common and effective dimension for many semantic tasks, offering a good balance between expressiveness and computational efficiency.

    **Required Library and Model Name:**

    *   **Library**: `google-generativeai` (Python client library for Google Gemini API)
    *   **Model Name for API calls**: `models/text-embedding-004` (or `text-embedding-004` depending on the client library version and usage pattern).

    *Note: Using this model will require a Google Cloud or Google AI Studio API key for authentication.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reasoning**:
    To use the selected Gemini embedding model, the first step is to install the required Python client library, `google-generativeai`.
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: google-generativeai !pip install -q google-generativeai
    print("Google Generative AI Python client library installation initiated.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reasoning**:
    To generate embeddings using the Gemini model, I need to configure the Google Generative AI client with an API key. This step initializes the client, allowing subsequent calls to the embedding model.
    """)
    return


@app.cell
def _():
    import os
    os.environ['GOOGLE_API_KEY'] = 'AIzaSyDrrYHqKCPG-Gr4SYe9IVtAumVc8YXL8jI'
    return


@app.cell
def _():
    import google.generativeai as genai
    import os

    # Configure Google Generative AI (Gemini) client using an environment variable.
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    if not GOOGLE_API_KEY:
        print("GOOGLE_API_KEY not set in environment. Skipping Google Generative AI configuration. Set GOOGLE_API_KEY to enable embeddings with Gemini.")


    genai.configure(api_key=GOOGLE_API_KEY)

    print("Google Generative AI client configured (api key sourced from env).")
    return (genai,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reasoning**:
    To generate embeddings for the 'Description' column, I will define a function that uses the configured Gemini model (`models/text-embedding-004`) to produce embeddings for each product description. I'll include a delay to manage potential rate limits and store the embeddings in a new column.
    """)
    return


@app.cell
def _(df_dataset, genai, time):
    from tqdm.auto import tqdm
    df_test = df_dataset.head(100).copy()
    df_test = df_test[df_test['Description'].str.strip().astype(bool)]
    # 1. Grab a small sample (TEST DATA ONLY)
    # We use .copy() to avoid SettingWithCopy warnings
    batch_size = 50
    texts = df_test['Description'].tolist()
    # 2. Clean data (Remove empty descriptions in the test set)
    embeddings = []
    print(f'Starting TEST run for {len(texts)} items...')
    # 3. Settings
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]
        try:
            response = genai.embed_content(model='models/text-embedding-004', content=batch, task_type='retrieval_document')
            embeddings.extend(response['embedding'])
            time.sleep(0.5)
    # 4. Loop through the small test data
        except Exception as e:
            print(f'Error at batch {i}: {e}')
            embeddings.extend([None] * len(batch))
    df_test['Description_Embedding'] = embeddings
    print("✅ Test Complete! Check 'df_test' for results.")
    # 5. Assign back to the TEST dataframe
    print(df_test['Description_Embedding'].head())
    return (tqdm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Task
    Generate embeddings for a sample of 5000 product descriptions from `df_dataset` using the `models/text-embedding-004` Gemini model. Implement concurrent processing with a `BATCH_SIZE` of 50, `MAX_WORKERS` of 2, and `MAX_RETRIES` of 3 to handle API calls efficiently and robustly. Store these embeddings in a new column named 'Description_Embedding' within the `df_sample` DataFrame.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate Embeddings for a Sample with Concurrent Processing and Retries

    ### Subtask:
    Generate embeddings for a sample of 5000 rows from `df_dataset` (after filtering out empty descriptions) using the `models/text-embedding-004` Gemini model. This will involve using `concurrent.futures` for parallel processing with a `BATCH_SIZE` of 50, `MAX_WORKERS` of 2, and `MAX_RETRIES` of 3, as specified. The embeddings will be stored in a new 'Description_Embedding' column in `df_sample`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reasoning**:
    To begin the concurrent embedding generation process, I need to import the `ThreadPoolExecutor` for parallel processing and define the constants for `SAMPLE_SIZE`, `BATCH_SIZE`, `MAX_WORKERS`, and `MAX_RETRIES` as specified in the instructions.
    """)
    return


@app.cell
def _():
    from concurrent.futures import ThreadPoolExecutor
    SAMPLE_SIZE = 5000
    BATCH_SIZE = 50
    # Define constants
    MAX_WORKERS = 2
    MAX_RETRIES = 3
    print('Constants for embedding generation defined.')
    return BATCH_SIZE, MAX_RETRIES, MAX_WORKERS, ThreadPoolExecutor


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reasoning**:
    To prepare the data for embedding, I will create a sample DataFrame `df_sample` from `df_dataset` by taking `SAMPLE_SIZE` unique product descriptions, ensuring they are not empty, and then reset its index.
    """)
    return


app._unparsable_cell(
    r"""
    df_dataset_clean = df_dataset[df_dataset['Description'].astype(bool)].copy()

    # Take a sample of unique descriptions
    if len(df_dataset_clean['Description'].unique()) > SAMPLE_SIZE:
        df_sample = df_dataset_clean.drop_duplicates(subset=['Description']).sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    else:
        df_sample = df_dataset_clean.drop_duplicates(subset=['Description']).reset_index(drop=True)


    print(f\"Created df_sample with {len(df_sample)} unique product descriptions.\")
    print(\"--- df_sample Head ---\")
    print(df_sample.
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reasoning**:
    To ensure robust embedding generation, I will define a function `embed_with_retry` that encapsulates the Gemini API call with retry logic, as specified in the instructions.
    """)
    return


@app.cell
def _(genai, time):
    def embed_with_retry(texts_to_embed, model_name, task_type, max_retries, retry_delay_seconds=5):
        retry_count = 0
        while retry_count <= max_retries:
            try:
                response = genai.embed_content(model=model_name, content=texts_to_embed, task_type=task_type)
                return response['embedding']
            except Exception as e:
                print(f'Error generating embeddings (Attempt {retry_count + 1}/{max_retries + 1}): {e}')
                retry_count = retry_count + 1
                if retry_count <= max_retries:
                    time.sleep(retry_delay_seconds)
                else:
                    print(f'Max retries reached for batch. Returning None for this batch.')
                    return None
    print('Defined embed_with_retry function.')  # Return None if all retries fail
    return (embed_with_retry,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Reasoning**:
    Now that the `embed_with_retry` function is defined and the `df_sample` is ready, I will prepare the descriptions, process them in batches using `ThreadPoolExecutor` with the defined retry logic, and store the generated embeddings in a new column in `df_sample`.
    """)
    return


@app.cell
def _(
    BATCH_SIZE,
    MAX_RETRIES,
    MAX_WORKERS,
    ThreadPoolExecutor,
    df_sample,
    embed_with_retry,
    tqdm,
):
    descriptions_to_embed = df_sample["Description"].astype(str).tolist()
    all_embeddings = [None] * len(descriptions_to_embed)
    # 5. Prepare descriptions from df_sample for embedding
    batches = []
    for i_1 in range(0, len(descriptions_to_embed), BATCH_SIZE):
        # 6. Initialize an empty list for all embeddings
        batches.append(
            (i_1, descriptions_to_embed[i_1 : i_1 + BATCH_SIZE])
        )  # Pre-allocate space for embeddings
    print(f"Total batches to process: {len(batches)}")
    # Prepare batches for concurrent processing
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_batch = {
            executor.submit(
                embed_with_retry,
                batch_texts,
                "models/text-embedding-004",
                "retrieval_document",
                MAX_RETRIES,
            ): (start_idx, batch_texts)
            for start_idx, batch_texts in batches
        }
        for future in tqdm(future_to_batch, desc="Generating embeddings"):
            start_idx, _ = future_to_batch[future]
            try:
                batch_embeddings = future.result()
                # 7. Use ThreadPoolExecutor for parallel processing
                if batch_embeddings is not None:
                    for j, emb in enumerate(batch_embeddings):
                        all_embeddings[start_idx + j] = emb
                else:
                    print(
                        f"Failed to get embeddings for batch starting at index {start_idx}"
                    )
            except Exception as exc:
                print(
                    f"Batch starting at index {start_idx} generated an exception: {exc}"
                )
    df_sample["Description_Embedding"] = all_embeddings
    print("\n--- df_sample Head with Embeddings ---")
    print(df_sample.head())
    print("\n--- df_sample Info with Embeddings ---")
    # 8. Add the all_embeddings list as a new column
    # 9. Print the head of df_sample and its info
    df_sample.info()  # Place embeddings in the correct position in all_embeddings  # Mark as failed if batch_embeddings is None (all retries failed)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
