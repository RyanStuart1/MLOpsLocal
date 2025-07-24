import openai
import os
import pandas as pd
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv
from pipeline.flow import drift_aware_pipeline
from pipeline.metrics import get_model_metrics
from pipeline.pdf_report import generate_pdf_report
import json
import textwrap

SYSTEM_PROMPT = textwrap.dedent("""\
You are a highly skilled credit risk model auditor writing for a regulatory audience.
**Do not apologize or mention inability to generate PDFs.**  
Whenever asked for a summary in PDF format, simply produce the summary text.
Answer user questions concisely and professionally. If they ask about drift or performance, refer to the context you’ve been given.
""")

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = openai.OpenAI(api_key=OPENAI_API_KEY)

def ask_openai(
    user_message: str,
    system_prompt: str,
    model: str,
    temperature: float = 1,
    max_tokens: int = 700,
    context: str = ""
) -> str:
    if not OPENAI_API_KEY:
        return "API key not set."
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": f"Here is some relevant context:\n{context}"},
                {"role": "user",   "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"OpenAI API error: {e}"
    
    
def show_chatbot_sidebar():

    drift_scores = {}
    tests        = {}
    metrics_map  = {}
    
    st.sidebar.markdown("### 💬 Ask AI for help")
    model_options = ["gpt-4", "gpt-3.5-turbo"]
    current_model = st.sidebar.selectbox(
        "Select LLM model",
        model_options,
        index=model_options.index("gpt-4")  # default GPT‑4
    )
    st.sidebar.info(f"Using model: `{current_model}`")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    uploaded = st.sidebar.file_uploader("Upload file here!", type=["json"])
    summary = None
    context = ""

    if uploaded is not None:
        # Only JSON for drift summaries
        try:
            text = uploaded.getvalue().decode("utf-8")
            summary = json.loads(text)
            st.sidebar.success("Uploaded JSON parsed")
        except Exception as e:
            st.sidebar.error(f"Could not parse upload: {e}")
            summary = None
    else:
        # no upload, try on-disk
        try:
            with open("artifacts/drift_summary.json", "r") as f:
                summary = json.load(f)
        except FileNotFoundError:
            summary = None

    try:
        shap_summary = json.load(open("artifacts/shap_summary.json", "r"))
        context += "\nSHAP Feature Comparison (top 5 by Δ):\n"
        for row in shap_summary[:5]:
            feature = row["feature"]
            v1      = row["mean_v1"]
            v2      = row["mean_v2"]
            d       = row["delta"]
            context += f" • {feature}: v1={v1:.3f}, v2={v2:.3f}, Δ={d:.3f}\n"
    except FileNotFoundError:
        # no SHAP file yet
        pass
    
    metrics = get_model_metrics()
    # Gather drift scores from metrics
    drift_scores = {}
    if summary:      
        for m in summary.get("metrics", []):
            mid = m["metric_id"]  # e.g. "ValueDrift(column=age)"
            if not mid.startswith("ValueDrift"):
                continue
            # pull out the text between "column=" and ")"
            feature = mid[mid.find("column=") + len("column=") : -1]
            drift_scores[feature] = m["value"]
    else:
        st.sidebar.warning("⚠️ Drift summary not found. Please run the pipeline or upload the file.")
        return

    # Gather test results from tests
    #    e.g. [{"metric_id":"KolmogorovSmirnovTest(column=age)", "value":{"p_value":0.02…}}, …]
    tests = {}
    for m in summary.get("metrics", []):
        if m.get("test"):
            feature = m["metric_id"].split("(column=")[1].rstrip(")")
            tests[feature] = (m["test"], None)

    # Overall drift count/share
    #    (still from metrics_map of DriftedColumnsCount)
    metrics_map = {m["metric_id"]: m["value"] for m in summary["metrics"]}

    overall_key = next((k for k in metrics_map if k.startswith("DriftedColumnsCount")), None)
    if overall_key:
        oc = metrics_map[overall_key]
        context += (
            f"Overall drift: {int(oc['count'])} of "
            f"{len(summary['metrics'])-1} features "
            f"({oc['share']*100:.1f}%).\n"
        )

    if metrics:
        context += f"Latest Metrics: Val Accuracy: {metrics['val_accuracy']}, "
        context += f"Val ROC AUC: {metrics['val_roc_auc']}, "
        context += f"Test Accuracy: {metrics['test_accuracy']}, "
        context += f"Test ROC AUC: {metrics['test_roc_auc']}.\n"

    # only uses user input after clicking send
    user_input = st.sidebar.text_input("Ask anything", key="sidebar_chat_input")
    send = st.sidebar.button("Send")

    if send and user_input:
        # Detect retraining intent
        lowered = user_input.lower()
        retrain_keywords = ["retrain", "drift check", "update model", "refresh model"]

        if any(word in lowered for word in retrain_keywords):
            with st.sidebar:
                with st.spinner("Running drift-aware pipeline..."):
                        result = drift_aware_pipeline()
                        if result["retrained"]:
                            reply = "Significant drift detected and the model was retrained."
                        else:
                            reply = "ℹ️ No retraining needed. Drift was within acceptable range."
        else:
            with st.sidebar:
                with st.spinner("Thinking..."):
                    raw_reply = ask_openai(
                        user_input,
                        system_prompt=SYSTEM_PROMPT, 
                        model=current_model, 
                        context=context
                    )
                    if "Here's the summary" in raw_reply:
                        reply = raw_reply.split("Here's the summary:", 1)[1].strip()
                    else:
                        reply = raw_reply

        # Build Executive summary
        overall_share = None
        for k, v in metrics_map.items():
            if k.startswith("DriftedColumnsCount"):
                overall_share = v["share"] * 100
                break

        # Format drift line safely
        if overall_share is None:
            share_str = "n/a"
        else:
            share_str = f"{overall_share:.1f}%"
        lines = [f"Overall drift detected across {len(drift_scores)} features: {share_str}."]

        # checks keys exists and appends the latest performance metrics
        if metrics and all(k in metrics for k in ["val_accuracy","val_roc_auc","test_accuracy","test_roc_auc"]):
            lines.append(
                "Model performance – "
                f"Val Acc: {metrics['val_accuracy']:.2f}, "
                f"Val ROC AUC: {metrics['val_roc_auc']:.2f}, "
                f"Test Acc: {metrics['test_accuracy']:.2f}, "
                f"Test ROC AUC: {metrics['test_roc_auc']:.2f}."
            )

        # Append the top 5 SHAP deltas from your JSON summary (if it exists)
        try:
            with open("artifacts/shap_summary.json") as f:
                shap_summary = json.load(f)
        except FileNotFoundError:
            shap_summary = []

        if shap_summary:
            lines.append("Top SHAP feature‐importance changes:")
            for row in shap_summary[:5]:
                lines.append(f" • {row['feature']}: Δ={row['delta']:.3f}")

        # Join into one multi‐line string
        summary_text = "\n".join(lines)

        # retraining recommendation 
        retrain_result = drift_aware_pipeline()
        if retrain_result["retrained"]:
            summary_text += (
                "\n\n**Retraining recommendation:** "
                "Significant drift detected; model retraining is recommended."
            )
        else:
            summary_text += (
                "\n\n**Retraining recommendation:** "
                "No retraining needed; drift was within acceptable range."
            )


        # Build the detailed drift_data list for all features
        drift_data = []

        # pick your test (KS vs Chi‑square) from your tests mapping
        for feature, score in drift_scores.items():
            test_name = tests.get(feature, ("UnknownTest", None))[0]
            is_drift = isinstance(score, (int, float)) and score < 0.05
            status = "Significant drift" if is_drift else "No significant drift"
            context += f"\n– **{feature}** drift {score:.3f}, test used: {test_name}\n"

            context += textwrap.dedent(f""" 
                Drift detected in **{feature}**:
                Drift score: {score:.3f}
                Test used: {test_name}
                Status: {status}
                """)
            
            drift_data.append({
                "feature":   feature,
                "score":     score,
                "test_used": test_name,
                "status":    status
            })

        # Generate the PDF using summary + table data
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_monitoring_report_{ts}.pdf"

        shap_data = []
        try:
            shap_data = json.load(open("artifacts/shap_summary.json"))
        except FileNotFoundError:
            pass

        pdf_path = generate_pdf_report(
            summary_text, 
            drift_data,
            shap_data=shap_data,
            filename=filename,
            chatgpt_summary=reply
        )

        with open(pdf_path, "rb") as f:
            st.sidebar.download_button(
                label="📄 Download AI Report as PDF",
                data=f,
                file_name=filename,
                mime="application/pdf"
            )

        st.session_state.chat_history.append((user_input, reply))

    if st.session_state.chat_history:
        for user_msg, bot_reply in reversed(st.session_state.chat_history[-5:]):
            st.sidebar.markdown(f"**You:** {user_msg}")
            st.sidebar.markdown(f"🧠 {bot_reply}")

    if st.sidebar.button("Trigger Drift Check & Retrain (Locally)"):
        with st.sidebar:
            with st.spinner("Running drift-aware pipeline..."):
                result = drift_aware_pipeline()
        if result["retrained"]:
            st.sidebar.success("Model retrained due to detected drift.")
        else:
            st.sidebar.info("ℹ️ No retraining needed. Drift within acceptable range.")
