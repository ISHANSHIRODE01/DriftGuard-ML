"""Streamlit monitoring dashboard reading directly from the drift store."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.db.repository import DriftRepository

st.set_page_config(page_title="DriftGuard Monitor", page_icon="📉", layout="wide")


@st.cache_resource
def get_repo() -> DriftRepository:
    return DriftRepository(os.getenv("DATABASE_URL"))


st.title("DriftGuard — Model Drift Monitor")
st.caption(
    "ELEC2 benchmark · KS + PSI drift detection · champion-challenger promotion gate"
)

repo = get_repo()

try:
    timeline = pd.DataFrame(repo.timeline())
    models = pd.DataFrame(repo.model_history())
    features = pd.DataFrame(repo.most_drifted_features())
except Exception as exc:
    st.error(f"Cannot read monitoring store: {exc}")
    st.info("Run `python experiments/run_experiment.py` to populate it.")
    st.stop()

if timeline.empty:
    st.warning("No monitoring runs recorded yet.")
    st.stop()

# ---------------------------------------------------------------- KPI row
champ = models[models.is_champion]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Batches monitored", len(timeline))
c2.metric("Models trained", len(models))
c3.metric(
    "Champion",
    f"v{int(champ.version.iloc[0])}" if not champ.empty else "none",
)
c4.metric("Promotions", int(timeline.retrain_triggered.sum()))

# ---------------------------------------------------- performance + drift
fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
    subplot_titles=("Batch performance of the serving model",
                    "Features drifted per batch"),
)
fig.add_trace(
    go.Scatter(x=timeline.batch_index, y=timeline.batch_accuracy,
               name="Accuracy", mode="lines+markers", line=dict(color="#1e8449")),
    row=1, col=1,
)
fig.add_trace(
    go.Scatter(x=timeline.batch_index, y=timeline.batch_f1,
               name="F1", mode="lines+markers", line=dict(color="#2980b9")),
    row=1, col=1,
)
fig.add_trace(
    go.Bar(x=timeline.batch_index, y=timeline.n_features_drifted,
           name="Drifted features",
           marker_color=["#e67e22" if d else "#95a5a6"
                         for d in timeline.dataset_drift_detected]),
    row=2, col=1,
)
for _, r in timeline[timeline.retrain_triggered].iterrows():
    fig.add_vline(x=r.batch_index, line_dash="dot",
                  line_color="#8e44ad", opacity=0.5, row=1, col=1)

fig.update_yaxes(title_text="Score", row=1, col=1)
fig.update_yaxes(title_text="Count", row=2, col=1)
fig.update_xaxes(title_text="Batch index", row=2, col=1)
fig.update_layout(height=620, hovermode="x unified",
                  legend=dict(orientation="h", y=1.08))
st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------ two columns
left, right = st.columns(2)

with left:
    st.subheader("Feature drift frequency")
    if not features.empty:
        st.bar_chart(features.set_index("feature")["drift_rate"])
        st.dataframe(features, use_container_width=True, hide_index=True)

with right:
    st.subheader("Model registry")
    st.dataframe(
        models[["version", "algorithm", "val_accuracy", "val_f1", "is_champion"]],
        use_container_width=True, hide_index=True,
    )

st.subheader("Retraining decisions")
events = timeline[timeline.retrain_triggered][
    ["batch_index", "n_features_drifted", "batch_accuracy", "trigger_reason"]
]
st.dataframe(events, use_container_width=True, hide_index=True)

with st.expander("How the retrain decision works"):
    st.markdown(
        """
**Trigger** — fires when *either* condition holds:

* **Distribution drift**: more than 50% of monitored features fail a
  Kolmogorov-Smirnov test at p < 0.05 **and** have PSI ≥ 0.10. The PSI floor
  is essential — with large samples KS flags trivial shifts as significant.
* **Performance breach**: batch accuracy falls more than 5 points below the
  champion's validation accuracy.

Monitoring only distributions misses concept drift where features look stable
but the feature→label relationship changed. Monitoring only performance means
waiting for labels, which in production arrive late or never.

**Promotion gate** — a challenger must beat the champion by ≥ 1 point on the
*same* unseen batch. Both are scored on identical data; comparing a fresh
validation score to a stale one would rig the contest. Retrains that fail the
gate are logged and discarded.
        """
    )
