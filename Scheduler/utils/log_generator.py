import time

def log_generator():
    """
    Generator function that yields log messages for streaming.
    """
    while True:
        # Simulate log message generation
        yield f"data: Log message at {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        time.sleep(1)  # Adjust the sleep time as needed