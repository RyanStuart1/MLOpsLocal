import streamlit as st
from pipeline.chatbot import show_chatbot_sidebar

st.set_page_config(page_title="Credit Risk App", layout="wide")

# Bootstrap Icons CDN + Styling
st.markdown("""
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css">
    <style>
        .card-container a {
            text-decoration: none !important;
            color: inherit !important;
        }
        .card-container a:hover {
            text-decoration: none !important;
        }
        .card-container a div {
            text-decoration: none !important;
            color: inherit !important;
        }

        .centered-text {
            text-align: center;
            margin-top: 2rem;
        }

        .card-container {
            display: flex;
            justify-content: center;
            gap: 3rem;
            margin-top: 3rem;
            flex-wrap: wrap;
        }

        .card {
            background-color: #1e1e1e;
            border-radius: 20px;
            padding: 3rem 2rem;
            width: 220px;
            height: 220px;
            text-align: center;
            color: white;
            cursor: pointer;
            transition: all 0.3s ease-in-out;
            box-shadow: 0 6px 16px rgba(0,0,0,0.3);
            border: 1px solid #333;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            font-size: 1.2rem;
        }

        .card:hover {
            transform: scale(1.08);
            border-color: #FF4B4B;
        }

        .card-icon-circle {
            background: linear-gradient(135deg, #ff6a00, #ff0000);
            border-radius: 50%;
            padding: 1rem;
            display: flex;
            align-items: center;
            justify-content: center;
            width: 70px;
            height: 70px;
            margin-bottom: 1rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            transition: box-shadow 0.3s ease;
        }

        .card-icon-circle i {
            font-size: 1.8rem;
            color: white;
        }

        .card:hover .card-icon-circle {
            box-shadow: 0 0 15px #ff6a00;
        }
            
        .intro-box {
        max-width: 720px;
        margin: 0 auto;
        padding: 2rem 2.5rem;
        background-color: #1e1e1e;
        border-radius: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 6px 16px rgba(0,0,0,0.3);
        }
        .intro-box h1 {
            margin-bottom: 1.5rem;
        }
        .intro-box ul {
            text-align: left;
            display: inline-block;
            margin-top: 0.5rem;
            margin-bottom: 1rem;
        }
        .gradient-icon {
        background: linear-gradient(135deg, #ff6a00, #ff0000);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
        }
        .gradient-icon:hover {
        filter: brightness(1.2);
        transform: scale(1.1);
        transition: all 0.5s ease;
        }
        .intro-box {
        animation: fadeIn 0.6s ease-in-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="intro-box">
    <h1>
        <i class="bi bi-credit-card-2-front-fill gradient-icon" style='margin-right: 10px;'></i>
        Welcome to the Credit Risk App
    </h1>
    <p>This dashboard helps you:</p>
    <ul>
        <li>Predict loan default risk</li>
        <li>View model performance metrics</li>
        <li>Explore insights about the dataset and model</li>
        <li>Batch Model monitoring</li>
    </ul>
    <p>Use the cards below or the sidebar to navigate between pages.</p>
</div>
""", unsafe_allow_html=True)

# Cards
st.markdown("""
<div class="card-container">
    <a href="/Predict" target="_self">
        <div class="card">
            <div class="card-icon-circle">
                <i class="bi bi-robot"></i>
            </div>
            <div>Predict</div>
        </div>
    </a>
    <a href="/Model_insights" target="_self">
        <div class="card">
            <div class="card-icon-circle">
                <i class="bi bi-graph-up"></i>
            </div>
            <div>Model Insights</div>
        </div>
    </a>
    <a href="/Model_monitoring" target="_self">
        <div class="card">
            <div class="card-icon-circle">
                <i class="bi bi-shield-check"></i>
            </div>
            <div>Model Monitoring</div>
        </div>
    </a>
    <!-- MLflow -->
    <a href="http://127.0.0.1:7000" target="_blank">
        <div class="card">
            <div class="card-icon-circle">
                <i class="bi bi-database-fill-gear"></i>
            </div>
            <div>MLflow</div>
        </div>
    </a>
    <!-- Prefect -->
    <a href="http://127.0.0.1:4200" target="_blank">
        <div class="card">
            <div class="card-icon-circle">
                <i class="bi bi-diagram-3-fill"></i>
            </div>
            <div>Prefect</div>
        </div>
    </a>
</div>
""", unsafe_allow_html=True)

show_chatbot_sidebar()
