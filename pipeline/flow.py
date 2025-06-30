from prefect import flow, task
from pipeline.train import train_model  # assume you wrap training in a function

@task
def run_training():
    return train_model()

@flow
def credit_risk_pipeline():
    run_training()

if __name__ == "__main__":
    credit_risk_pipeline()
