import streamlit as st

def render_home_button():
    st.markdown("""
        <link rel="stylesheet"
              href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css">
        <style>
        .top-right-home {
            display: flex;
            justify-content: flex-end;
            margin-bottom: 1rem;
        }
        .home-card {
            background-color: #1e1e1e;
            border-radius: 15px;
            padding: 1rem 1.5rem;
            text-align: center;
            color: white !important;
            text-decoration: none !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            border: 1px solid #333;
            display: inline-flex;
            align-items: center;
            gap: 0.6rem;
            transition: all 0.2s ease-in-out;
        }
        .home-card-icon {
            background: linear-gradient(135deg, #ff6a00, #ff0000);
            border-radius: 50%;
            padding: 0.6rem;
            display: flex;
            align-items: center;
            justify-content: center;
            width: 40px;
            height: 40px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.25);
        }
        .home-card-icon i {
            color: white;
            font-size: 1.1rem;
        }
        .home-card:hover {
            transform: scale(1.05);
            border-color: #ff6a00;
        }
        </style>

        <div class="top-right-home">
            <a href="/" target="_self" class="home-card">
                <div class="home-card-icon">
                    <i class="bi bi-house-door-fill"></i>
                </div>
                <div>Home</div>
            </a>
        </div>
    """, unsafe_allow_html=True)
