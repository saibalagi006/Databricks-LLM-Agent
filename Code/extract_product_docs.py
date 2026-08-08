"""
Databricks notebook to extract text from PDF files iteratively and store in Unity Catalog table.
Processes PDFs in batches to handle large volumes efficiently (~500 files).
"""

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
import os
import tempfile
import PyPDF2

# Initialize Spark Session
spark = SparkSession.builder.appName("PDFTextExtraction").getOrCreate()

# Configuration
VOLUME_PATH = "/Volumes/llmagent/dev/data_volume/01_Data_Files/product_docs"
CATALOG = "llmagent"
SCHEMA = "dev"
TABLE_NAME = "product_details"
FULL_TABLE_NAME = f"{CATALOG}.{SCHEMA}.{TABLE_NAME}"
BATCH_SIZE = 50  # Process 50 PDFs at a time to manage memory

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file using PyPDF2.

    Args:
        file_path: Path to the PDF file

    Returns:
        Extracted text content
    """
    try:
        text_content = []
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                if text:
                    text_content.append(text)
        return "\n".join(text_content)
    except Exception as e:
        print(f"Error extracting text from {file_path}: {str(e)}")
        return ""

def get_product_name(filename: str) -> str:
    """Extract product name from filename (without extension)."""
    return os.path.splitext(filename)[0]

def process_pdfs_batch(batch_files: list) -> list:
    """
    Process a batch of PDF files and extract text.

    Args:
        batch_files: List of file info objects for the batch

    Returns:
        List of tuples (product_name, product_doc)
    """
    batch_data = []

    for file_info in batch_files:
        file_path = file_info.path
        filename = file_info.name

        # Create temporary local copy and extract text
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            try:
                # Copy file from volume to temporary location
                dbutils.fs.cp(file_path, f"file:{tmp_file.name}", recurse=False)

                # Extract text
                text_content = extract_text_from_pdf(tmp_file.name)
                product_name = get_product_name(filename)

                if text_content.strip():
                    batch_data.append((product_name, text_content))
                    print(f"  ✓ {filename}: {len(text_content)} chars")
                else:
                    print(f"  ⚠ {filename}: No text extracted")

            except Exception as e:
                print(f"  ✗ {filename}: {str(e)}")
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_file.name)
                except:
                    pass

    return batch_data

def create_table_if_not_exists():
    """Create the target table if it doesn't exist."""
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {FULL_TABLE_NAME} (
            product_name STRING,
            product_doc STRING
        )
        USING DELTA
    """)

def process_all_pdfs():
    """Main function to process all PDFs iteratively in batches."""

    # Check if volume path exists
    try:
        files = dbutils.fs.ls(VOLUME_PATH)
    except Exception as e:
        print(f"Error accessing volume path: {str(e)}")
        return

    # Filter for PDF files only
    pdf_files = [f for f in files if f.name.lower().endswith('.pdf')]

    if not pdf_files:
        print(f"No PDF files found in {VOLUME_PATH}")
        return

    total_files = len(pdf_files)
    print(f"Found {total_files} PDF files to process")
    print(f"Processing in batches of {BATCH_SIZE}\n")

    # Create table if not exists
    create_table_if_not_exists()

    # Process files in batches
    total_processed = 0
    total_succeeded = 0

    for batch_start in range(0, total_files, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, total_files)
        batch_files = pdf_files[batch_start:batch_end]
        batch_num = (batch_start // BATCH_SIZE) + 1
        total_batches = (total_files + BATCH_SIZE - 1) // BATCH_SIZE

        print(f"\n--- Batch {batch_num}/{total_batches} (Files {batch_start + 1}-{batch_end}) ---")

        # Process batch
        batch_data = process_pdfs_batch(batch_files)
        total_processed += len(batch_files)
        total_succeeded += len(batch_data)

        if batch_data:
            # Create DataFrame from batch
            schema = StructType([
                StructField("product_name", StringType(), True),
                StructField("product_doc", StringType(), True)
            ])

            df_batch = spark.createDataFrame(batch_data, schema=schema)

            # Append to table (using append mode for all batches except first)
            mode = "overwrite" if batch_start == 0 else "append"
            df_batch.write.mode(mode).option("mergeSchema", "true").saveAsTable(FULL_TABLE_NAME)

            print(f"  → Batch written ({len(batch_data)} rows)")
        else:
            print(f"  → No data to write from this batch")

    # Final summary
    print(f"\n{'='*60}")
    print(f"Processing Complete!")
    print(f"{'='*60}")
    print(f"Total files processed: {total_processed}")
    print(f"Successfully extracted: {total_succeeded}")
    print(f"Failed: {total_processed - total_succeeded}")

    # Verify final results
    try:
        row_count = spark.sql(f"SELECT COUNT(*) as count FROM {FULL_TABLE_NAME}").collect()[0]['count']
        print(f"\nFinal table row count: {row_count}")

        # Show sample
        print(f"\nSample data (first 3 rows):")
        spark.sql(f"""
            SELECT
                product_name,
                LENGTH(product_doc) as text_length,
                SUBSTR(product_doc, 1, 100) as preview
            FROM {FULL_TABLE_NAME}
            LIMIT 3
        """).show(truncate=False)

    except Exception as e:
        print(f"Error retrieving final results: {str(e)}")

if __name__ == "__main__":
    process_all_pdfs()
