# M.Tech Talent Dashboard

This repository contains the source code for an interactive data analytics dashboard built with Streamlit, designed to visualize student placement data for the M.Tech program at MPSTME. The dashboard provides key insights into academic performance, professional experience, and their correlation, helping to streamline the placement process and identify key trends.

The project is built with a focus on modern software development practices, including a robust CI/CD pipeline and Dockerization for a reproducible and reliable deployment process.

-----

### Features

  * **Interactive Dashboard**: Provides a dynamic and easy-to-use interface to explore student data.
  * **Key Metrics**: Displays high-level performance highlights such as total candidates, average CGPA, and percentages of students with internship or full-time experience.
  * **Academic Insights**: Visualizes student academic performance through CGPA distribution plots and breakdowns by program (AI/DS).
  * **Professional Experience Analytics**: Presents data on internship participation, full-time experience, and popular roles.
  * **Correlation Analysis**: Explores the relationship between academic performance and professional experience.
  * **Reproducible Environment**: Uses a `requirements.txt` and `Dockerfile` to ensure the application can be run consistently anywhere.

-----

### Technologies Used

  * **Python**: The core programming language.
  * **Streamlit**: For building the interactive web application.
  * **Pandas**: For data manipulation and analysis.
  * **Matplotlib & Seaborn**: For creating data visualizations.

-----

### How to Run Locally

To run this application on your local machine, you only need to have [Docker](https://www.docker.com/get-started/) installed. The project's `Dockerfile` handles all other dependencies.

1.  **Clone the Repository**:

    ```bash
    git clone https://github.com/AtharSayed/PlcementAnalysis.git
    cd PlcmentAnalysis
    ```

2.  **Build the Docker Image**:

    ```bash
    docker build -t placement-analysis .
    ```

3.  **Run the Docker Container**:

    ```bash
    docker run -p 8501:8501 placement-analysis
    ```

After running the command, open your web browser and navigate to `http://localhost:8501`.

-----

### CI/CD Pipeline

This project is configured with a Continuous Integration (CI) and Continuous Deployment (CD) pipeline using GitHub Actions.

  * **Continuous Integration (CI)**: A workflow is triggered on every push to the `main` branch. It automatically checks for linting errors, installs dependencies, and ensures the Docker image builds successfully. This guarantees the code is always in a working, deployable state.
  * **Continuous Deployment (CD)**: The pipeline is also set up to automatically push the validated and built Docker image to Docker Hub. This makes the latest version of the application easily accessible and ready for deployment to any environment that supports Docker.

The live application can be viewed on Streamlit Cloud, which is configured for automatic deployments from this repository:

[https://mpstme-mtech-talent-analytics.streamlit.app/](https://mpstme-mtech-talent-analytics.streamlit.app/)
