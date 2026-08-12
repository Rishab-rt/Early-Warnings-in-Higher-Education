# Early-Warnings-in-Higher-Education 🎓

Predicts STEM student outcomes (Dropout / Enrolled / Graduate) to flag at-risk students early. Trains Logistic Regression and Decision Tree models, evaluates them with confusion matrices and SHAP feature importance, and serves an interactive risk predictor via Streamlit.

[![Stars](https://img.shields.io/github/stars/Rishab-rt/Early-Warnings-in-Higher-Education?style=flat-square)](https://github.com/Rishab-rt/Early-Warnings-in-Higher-Education/stargazers)
[![Forks](https://img.shields.io/github/forks/Rishab-rt/Early-Warnings-in-Higher-Education?style=flat-square)](https://github.com/Rishab-rt/Early-Warnings-in-Higher-Education/forks)

## Table of Contents 📚

- [Project Overview](#project-overview)
- [Features](#features-✨)
- [Tech Stack](#tech-stack-💻)
- [Installation](#installation--)
- [Usage](#usage-🚀)
- [Project Structure](#project-structure-📂)
- [Contributing](#contributing--)
- [License](#license-📄)
- [Important Links](#important-links-🔗)
- [Footer](#footer-✨)

## Project Overview 📝

This project aims to predict the academic success and potential dropout risks of STEM students. By analyzing various student demographic and academic performance indicators, it trains machine learning models to forecast whether a student is likely to drop out, remain enrolled, or graduate. The system provides an interactive Streamlit application for exploring the data, auditing data quality, and predicting individual student outcomes. It also includes a robust evaluation of the models using confusion matrices and SHAP feature importance, with a special focus on an 'early warning' system using only first-semester data.

## Features ✨

- **Student Outcome Prediction:** Predicts student status (Dropout, Enrolled, Graduate) using Logistic Regression and Decision Tree models.
- **Early Warning System:** Identifies at-risk students based on first-semester academic performance.
- **Interactive Streamlit Dashboard:** Provides an intuitive interface for data exploration, auditing, and risk prediction.
- **Data Auditing & Cleaning:** Implements comprehensive data quality checks and cleaning procedures.
- **Model Evaluation:** Detailed analysis of model performance using classification reports, confusion matrices, and SHAP feature importance.
- **Containerization:** Docker support for easy deployment and reproducibility.
- **Feature Engineering:** Creates new features from existing data to improve model performance (e.g., approval ratios, grade deltas).

## Tech Stack 💻

- **Languages:** Python
- **Frameworks/Libraries:**
  - Streamlit: For building the interactive web application.
  - Scikit-learn: For machine learning model training and evaluation.
  - Pandas: For data manipulation and analysis.
  - NumPy: For numerical operations.
  - Matplotlib & Seaborn: For data visualization.
  - Joblib: For saving and loading trained models.
  - SHAP: For model interpretability and feature importance analysis.
- **Containerization:** Docker

## Installation 🛠️

**Prerequisites:**
- Python 3.10+ recommended.

**Steps:**
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Rishab-rt/Early-Warnings-in-Higher-Education.git
   cd Early-Warnings-in-Higher-Education
   ```

2. **Set up a virtual environment (recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows use: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the models (required before running app/evaluation):**
   The `.pkl` model artifacts are gitignored. You need to train them locally.
   ```bash
   python model.py
   ```
   This will create `logistic_model.pkl`, `decision_tree_model.pkl`, and `scaler.pkl` in the root directory.

## Usage 🚀

### Running the Streamlit Application 🏃

After setting up and training the models, you can launch the interactive Streamlit application:

```bash
streamlit run app.py
```

This will open the application in your browser, allowing you to:
- **Data Overview:** Explore Exploratory Data Analysis (EDA) and visualize relationships between student attributes and outcomes.
- **Data Audit:** Review the data cleaning process, check for issues like missing values, duplicates, and class imbalance.
- **Risk Predictor:** Input student metrics to get a predicted outcome probability using the trained Logistic Regression model.
- **Model Evaluation:** Understand the model's performance and the key factors influencing predictions through SHAP importance plots and confusion matrices.

### Running Evaluation Scripts 📊

To generate detailed performance reports and visualizations (confusion matrices, SHAP plots), run:

```bash
python evaluate.py
```

This script will output classification reports to the console and save plots to the `plots/` directory.

### Data Auditing Script 🧐

To run a quick audit of the data cleaning process and print a report to the console:

```bash
python audit_data.py
```

### Docker Usage 🐳

To build and run the application using Docker:

1. **Build the Docker image:**
   ```bash
   docker build -t early-warnings-app .
   ```

2. **Run the Docker container:**
   ```bash
   docker run -p 8501:8501 early-warnings-app
   ```
   The application will be accessible at `http://localhost:8501`.

Alternatively, use `docker-compose`:

```bash
docker-compose up --build
```

## Project Structure 📂

```
Early-Warnings-in-Higher-Education/
├── data/
│   └── data.csv
├── plots/
├── .streamlit/
│   └── config.toml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── app.py
├── audit_data.py
├── data_quality.py
├── evaluate.py
├── model.py
└── DEPLOYMENT.md
```

- `data/`: Contains the raw dataset.
- `plots/`: Generated visualizations (confusion matrices, SHAP plots).
- `.streamlit/`: Streamlit configuration files.
- `Dockerfile`: Instructions for building the Docker image.
- `docker-compose.yml`: Docker Compose configuration for running the app.
- `requirements.txt`: Project dependencies.
- `app.py`: The main Streamlit application file.
- `audit_data.py`: Script for running data quality checks.
- `data_quality.py`: Core data loading, cleaning, and auditing functions.
- `evaluate.py`: Script for model evaluation and generating plots.
- `model.py`: Script for training and saving the machine learning models.
- `DEPLOYMENT.md`: (Currently empty) Potentially for deployment instructions.

## How to use 🤔

This project is designed to help educational institutions proactively identify students who may be at risk of dropping out. By leveraging historical student data, the system provides:

1.  **Predictive Insights:** Forecasts the likelihood of a student graduating, remaining enrolled, or dropping out based on their academic and demographic profile.
2.  **Early Intervention:** The 'Early Warning System' feature allows institutions to identify students in need of support early in their academic journey (using only first-semester data).
3.  **Data-Driven Decisions:** Provides tools for understanding data quality, the factors influencing student success, and the performance of predictive models.
4.  **Interactive Exploration:** The Streamlit dashboard enables users to explore the data, test hypothetical student scenarios, and understand model predictions visually.

**Real-world Use Case:**
An academic advisor can use the **Risk Predictor** in the Streamlit app to input the details of a struggling student. The application will output the probabilities of different outcomes, helping the advisor understand the severity of the risk and tailor interventions. For example, if a student has a high probability of dropping out, the advisor can then reach out with targeted academic support, counseling, or financial aid information.

## Contributing 🤝

Contributions are welcome! Please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

Please ensure your code adheres to the project's style and includes tests where applicable.

## License 📄

This project is not explicitly licensed. Please refer to the repository for details.

## Important Links 🔗

- **Repository:** [Early-Warnings-in-Higher-Education](https://github.com/Rishab-rt/Early-Warnings-in-Higher-Education)

## Footer ✨

Developed with ❤️ by Rishab-rt.

**Repository:** [Early-Warnings-in-Higher-Education](https://github.com/Rishab-rt/Early-Warnings-in-Higher-Education)

**Author:** Rishab-rt

**Contact:** [Contact the Author](mailto:your_email@example.com) (Replace with actual contact if available)

--- 

*Feel free to ⭐ star, 🍴 fork, and 📣 raise issues on the repository!* 


---
