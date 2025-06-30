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

>[!IMPORTATNT]
>Below are useful commands for running MLFLOW UI, PREFECT and STREAMLIT
>
>```bash
>mlflow ui --port 7000
>
>prefect server start
>
>streamlit run dashboard/app.py
>```

```
MlOpsLocal/
├── .venv/                 # Virtual environment
│ 
├── dashboard/             # Streamlit dashboard
│ └── app.py
│ 
├── data/                  # CSV datasets
│ └── data.csv
│ 
├── mlruns/                # MLflow tracking directory (auto-generated)
│ 
├── models/                # Trained model artifacts
│ └── Models.py   
│     
├── pipeline/              # Core ML pipeline scripts
│ ├── flow.py              # Prefect orchestration
│ ├── train.py             # Model training
│ ├── evaluate.py          # Model evaluation
│ └── preprocess.py        # Feature engineering
│ 
├── .gitignore             # Git ignore rules
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
```