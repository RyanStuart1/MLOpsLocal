# MLOpsLocal

>[!NOTE]
>
>## Useful Commands for Setup & Usage
>
>To get started with this MLOps pipeline, here are the essential commands:
>
>###
>
>###  Set up the environment
>```bash
>python -m venv .venv  
>.venv\Scripts\activate on Windows
>pip install -r requirements.txt
>```

>[!CAUTION]
>If flow.py fails to run, simply run the script from the root.
>### Commands to run pipeline
>```bash
>set PYTHONPATH=%cd%
>python pipeline\flow.py
>```

>[!IMPORTANT]
>Below are useful commands for running MLFLOW UI, PREFECT and STREAMLIT
>
>```bash
>mlflow ui --port 7000
>
>prefect server start
>
>streamlit run streamlit_app.py
>```

> ## Prefect Deployment
> The snippet below is used in ps1 to follow by "N" to use local storage and "Y" to create a yaml for deployment.
>```bash
>prefect deploy .\pipeline\flow.py:credit_risk_pipeline `
>  --name weekly-credit-risk `
>  --cron "0 9 * * 1" `
>  --timezone "UTC" `
>  --pool default
>```
> Verify the correct repository is being used for the pull request
> ```bash
> pull:
> - prefect.deployments.steps.pull_from_git:
>    repo: https://github.com/RyanStuart1/MLOpsLocal.git
>    branch: main
>    directory: .
>```
> Prefect workers are used to execute scheduled tasks
>```bash
>prefect worker start --pool default
>```

> ## Project Structure
>```
>MlOpsLocal/
>├── .venv/                 # Virtual environment
>├── pages/                 # Streamlit dashboard
>│   ├── 1_Predict.py
>│   ├── 2_Model_insights.py
>│   └── 3_Model_monitoring.py
>├── data/                  # CSV datasets
>│   └── data.csv
>├── mlruns/                # MLflow tracking directory
>├── models/                # Trained model artifacts
>├── pipeline/              # Core pipeline scripts
>│   ├── flow.py            # Prefect orchestration
>│   ├── train.py           # Model training
>│   ├── evaluate.py        # Model evaluation
>│   ├── preprocess.py      # Feature engineering
>│   └── model_drift.py     # Drift monitoring
>├── .gitignore             # Git ignore rules
>├── prefect.yaml           # Prefect deployment config
>├── README.md              # Project documentation
>├── requirements.txt       # Python dependencies
>└── streamlit_app.py       # Streamlit app host
>```