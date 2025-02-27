# Use a small Python base image
FROM python:3.10-slim

# Install FFmpeg
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Set a working directory
WORKDIR /app

# Copy requirements.txt first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# Expose the port that Flask/Gunicorn will run on
EXPOSE 5000

# Command to start your app (Gunicorn recommended for production)
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
