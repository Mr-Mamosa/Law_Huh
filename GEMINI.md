frogive me code into code txt file by replacing the previous code.
The backend framework has been switched from FastAPI/Flask to Django.
The `QueryEngine` logic is now part of a Django app named `rag_api`.
The frontend is built and served by the Django backend.
The `requirements.txt` has been updated to reflect the new dependencies.
The API calls in the frontend have been adjusted to use relative paths.

To run the application:

1.  **Install Python Dependencies:**
    Make sure you are in the root directory of the project (e.g., `/home/adnan/Law AI/`).
    ```bash
    pip install -r requirements.txt
    ```

2.  **Navigate to the `backend` directory:**
    ```bash
    cd backend
    ```

3.  **Run Django Migrations:**
    ```bash
    python manage.py migrate
    ```

4.  **Start the Django Development Server:**
    ```bash
    python manage.py runserver 0.0.0.0:8000
    ```
    The server will start and the `QueryEngine` will initialize. This might take some time (up to a minute or more depending on your system specifications) as it loads the necessary models. You will see output related to "Loading Legal AI Engine...".

5.  **Access the Application:**
    Once the Django server is running, open your web browser and go to `http://localhost:8000/`. The React frontend should be served.

    The API endpoints are:
    *   `http://localhost:8000/api/ask` (POST request)
    *   `http://localhost:8000/api/status` (GET request)

    You can now interact with the Law AI Advisor through the web interface.
