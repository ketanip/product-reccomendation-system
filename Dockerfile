# Use official slim Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install basic dependencies and OS utilities
RUN apt-get update && apt-get install -y \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY train.py .
COPY data.csv .
COPY filtered_data.csv .
COPY preprocessor.pkl .
COPY svd_model.pkl .
COPY X_reduced.npy .

# Optional: Run training only if model doesn't exist
RUN python3 -c "import os; import pathlib; pathlib.Path('X_reduced.npy').exists() or __import__('train')"


# Expose Streamlit port
EXPOSE 8501

# Start Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
