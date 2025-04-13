# uvicorn_config.py

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "main:app",  # Replace "main" with your FastAPI filename (without .py)
        host="localhost",
        port=8000,
        reload=True,  # Auto-reload in development
        # workers=4  # Adjust based on your CPU cores
    )
