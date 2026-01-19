import multiprocessing

# Bind to all interfaces on port 8000
bind = "0.0.0.0:8000"

# Dynamically calculate the number of workers based on CPU cores
# Formula: (2 * cores) + 1
workers = (multiprocessing.cpu_count() * 2) + 1

# Use the Uvicorn worker class for FastAPI compatibility
worker_class = "uvicorn.workers.UvicornWorker"

# Logging levels
loglevel = "info"
accesslog = "-"
errorlog = "-"