from pipeline.rag_utils import get_rag_context
from pipeline.metrics import get_model_metrics
from pipeline.pdf_report import generate_pdf_report
from pipeline.flow import drift_aware_pipeline

import json, os, textwrap
from datetime import datetime

SYSTEM_PROMPT = textwrap.dedent("""
You are a highly skilled credit risk model auditor writing for a regulatory audience.

Use all available information to answer accurately — including PDF reference documents (if available), data drift summaries, SHAP feature comparisons, and performance metrics.

When relevant, prioritize factual details grounded in reference documents (e.g., PDFs or reports). Otherwise, rely on your own expertise and the provided context.

**Do not apologize or mention inability to generate PDFs.**  
Whenever asked for a summary in PDF format, simply produce the summary text.

Answer user questions concisely and professionally. If they ask about drift or performance, refer to the context you’ve been given.
""")

def build_context():
    context = ""
    summary = None

    try:
        summary = json.load(open("artifacts/drift_summary.json"))
    except FileNotFoundError:
        pass

    try:
        shap_summary = json.load(open("artifacts/shap_summary.json"))
        context += "\nSHAP Feature Comparison (top 5 by Δ):\n"
        for row in shap_summary[:5]:
            context += f" • {row['feature']}: v1={row['mean_v1']:.3f}, v2={row['mean_v2']:.3f}, Δ={row['delta']:.3f}\n"
    except FileNotFoundError:
        shap_summary = []

    try:
        metrics = get_model_metrics()
        context += (
            f"Latest Metrics: Val Accuracy: {metrics['val_accuracy']:.3f}, "
            f"Val ROC AUC: {metrics['val_roc_auc']:.3f}, "
            f"Test Accuracy: {metrics['test_accuracy']:.3f}, "
            f"Test ROC AUC: {metrics['test_roc_auc']:.3f}.\n"
        )
    except Exception:
        metrics = {}

    if summary:
        for m in summary.get("metrics", []):
            if m["metric_id"].startswith("DriftedColumnsCount"):
                drift_count = int(m["value"]["count"])
                drift_share = m["value"]["share"]
                context += f"Overall drift: {drift_count} features ({drift_share*100:.1f}%).\n"

    return context.strip(), summary, metrics

def respond_to_query(user_input: str, model: str, system_prompt: str, context: str, rag_db=None):
    from pipeline.openai_helper import ask_openai

    rag_context = ""
    rag_keywords = ["rag", "pdf", "document", "report", "reference"]
    lowered = user_input.lower()
    use_rag = any(word in lowered for word in rag_keywords)

    if use_rag and rag_db is not None:
        try:
            rag_context = get_rag_context(user_input, rag_db)
        except Exception as e:
            rag_context = f"[RAG Error: {e}]"

    combined_context = f"{rag_context}\n\n{context}".strip()

    reply = ask_openai(
        user_message=user_input,
        system_prompt=system_prompt,
        model=model,
        context=combined_context
    )

    if "summary" in lowered or "pdf" in lowered or "report" in lowered:
        try:
            context, summary_json, metrics = build_context()

            # Load SHAP summary
            shap_summary = json.load(open("artifacts/shap_summary.json")) if os.path.exists("artifacts/shap_summary.json") else []

            # Load Drift summary
            drift_summary = json.load(open("artifacts/drift_summary.json")) if os.path.exists("artifacts/drift_summary.json") else {"metrics": []}

            # Extract drift_scores and tests from summary
            drift_scores = {}
            tests = {}

            for m in drift_summary.get("metrics", []):
                mid = m.get("metric_id", "")
                if mid.startswith("ValueDrift"):
                    feature = mid.split("column=")[1].rstrip(")")
                    val = m.get("value")
                    score = val.get("drift_score") if isinstance(val, dict) else val
                    drift_scores[feature] = score
                    tests[feature] = (m.get("test", "UnknownTest"), None)

            # Build drift_data list for PDF
            drift_data = []
            for feature, score in drift_scores.items():
                test_name = tests.get(feature, ("UnknownTest", None))[0]
                is_drift = isinstance(score, (int, float)) and score < 0.05
                status = "Significant drift" if is_drift else "No significant drift"
                drift_data.append({
                    "feature": feature,
                    "score": score,
                    "test_used": test_name,
                    "status": status
                })

            # Build executive summary text
            lines = []
            for m in summary_json.get("metrics", []):
                if m["metric_id"].startswith("DriftedColumnsCount"):
                    drift_count = int(m["value"]["count"])
                    drift_share = m["value"]["share"]
                    lines.append(f"Overall drift detected across {drift_count} features: {drift_share*100:.1f}%.")

            if metrics:
                lines.append(
                    "Model performance – "
                    f"Val Acc: {metrics['val_accuracy']:.2f}, "
                    f"Val ROC AUC: {metrics['val_roc_auc']:.2f}, "
                    f"Test Acc: {metrics['test_accuracy']:.2f}, "
                    f"Test ROC AUC: {metrics['test_roc_auc']:.2f}."
                )

            if shap_summary:
                lines.append("Top SHAP feature-importance changes:")
                for row in shap_summary[:5]:
                    lines.append(f" • {row['feature']}: Δ={row['delta']:.3f}")

            retrain_result = drift_aware_pipeline()
            if retrain_result["retrained"]:
                lines.append("**Retraining recommendation:** Significant drift detected; model retraining is recommended.")
            else:
                lines.append("**Retraining recommendation:** No retraining needed; drift was within acceptable range.")

            summary_text = "\n".join(lines)

            # Save PDF
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_monitoring_report_{ts}.pdf"
            pdf_path = generate_pdf_report(
                summary=summary_text,
                drift_data=drift_data,
                shap_data=shap_summary,
                filename=filename,
                chatgpt_summary=reply
            )
            return reply, pdf_path
        except Exception as e:
            return f"{reply}\n\n[PDF generation failed: {e}]", None

    return reply, None
