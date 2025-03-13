## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd fastapi-azure-api
   ```

2. **Create a virtual environment:**
   ```bash
   
   pyenv versions
   pyenv install 3.12.8
   pyenv local 3.12.8
   python -m venv .venv
   source .venv/bin/activate
   ```


3. **Install dependencies:**
   ```
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```
   docker compose up --build
   ```
