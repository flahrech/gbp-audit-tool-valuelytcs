from waitress import serve
from main import app

if __name__ == "__main__":
    print("Server starting on http://localhost:8000")
    serve(app, host="0.0.0.0", port=8000)