import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
from PIL import Image
import datetime

# Set page config
st.set_page_config(page_title="SKT DRAM Price Causality Visualization Dashboard", layout="wide")

# Use a narrow column to place logo and title on the top-left
col1, col2 = st.columns([2, 8])  # col1: logo + title, col2: empty or content later

with col1:
    st.image("impactiveAI_logo.png", width=1000)
st.markdown(
    "<h2 style='text-align: center; margin-top: 50px;'>SKT DRAM Price Causality Visualization Dashboard</h2>",
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# 1. DATA UPLOAD (INLINE + SMALLER)
# -----------------------------------------------------------------------------
st.markdown("### 📂 Upload Files")
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    ts_file = st.file_uploader("Aggregated Result", type=["csv"])
with col2:
    q_file = st.file_uploader("Quarterly ASP", type=["csv"])
with col3:
    graph_file = st.file_uploader("Causal Graph", type=["csv"])

@st.cache_data(show_spinner="Loading time-series …")
def load_ts(file):
    df = pd.read_csv(file)
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data(show_spinner="Loading quarterly …")
def load_quarterly(file):
    df = pd.read_csv(file)
    df["dt"] = pd.to_datetime(df["dt"])
    return df

if ts_file is None or q_file is None:
    st.info("⬆️  Upload both CSV files to proceed.")
    st.stop()

try:
    df_ts = load_ts(ts_file)
    df_q  = load_quarterly(q_file)
except Exception as e:
    st.error(f"❌  Failed to load: {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 2. TARGET DATE & VARIABLE SELECTION
# -----------------------------------------------------------------------------
min_date, max_date = df_ts["date"].min(), df_ts["date"].max()
target_dt = st.date_input("Select target date", value=datetime.date(2022, 10, 3),
                          min_value=min_date, max_value=max_date)

n_vars = st.slider("Top-N variables (by |value|)", 1, 10, 3)

latest_slice = (
    df_ts[df_ts["date"] == pd.to_datetime(target_dt)]
      .drop(columns=["date"])
      .T
      .reset_index()
)
latest_slice.columns = ["variable", "value"]
latest_slice["abs_value"] = latest_slice["value"].abs()
col_target_df = (latest_slice
                 .sort_values("abs_value", ascending=False)
                 .head(n_vars)
                 .reset_index(drop=True))
col_target = col_target_df["variable"].tolist()

st.markdown("### 🎯 Key Features with Strongest Causal Impact on Target")
st.dataframe(col_target_df[["variable", "value"]],
             use_container_width=True)

# 👉  pull real coefficients here
coef_input = dict(zip(col_target_df["variable"], col_target_df["value"]))

# -----------------------------------------------------------------------------
# 3. VISUALIZATION 1 – CAUSAL GRAPH
# -----------------------------------------------------------------------------
st.markdown("## 1️⃣ Causal Knowledge Graph")

if graph_file is not None:
    try:
        graph_db = pd.read_csv(graph_file)
        graph_db["dt"] = pd.to_datetime(graph_db["dt"])
        causal_table = graph_db[graph_db["dt"] == pd.to_datetime(target_dt)]

        if causal_table.empty:
            st.info("No causal links for the selected date.")
        else:
            G = nx.DiGraph()
            edge_colors, edge_weights, spring_w = [], [], {}

            for _, row in causal_table.iterrows():
                src = f"{row['source']}(t-{row['lag']})"
                tgt = f"{row['target']}(t)"
                eff = row["Causal_effect"]
                lag = row["lag"]

                G.add_edge(src, tgt)
                edge_colors.append("blue" if eff > 0 else "red")
                edge_weights.append(abs(eff) * 2)
                spring_w[(src, tgt)] = 1 / lag

            nx.set_edge_attributes(G, spring_w, "weight")
            pos = nx.spring_layout(G, seed=42, weight="weight")

            fig_graph = plt.figure(figsize=(12, 8))
            nx.draw(
                G, pos,
                with_labels=True,
                node_color="lightblue",
                edge_color=edge_colors,
                width=edge_weights,
                node_size=1000,
                font_size=9,
                arrowsize=12,
                connectionstyle="arc3,rad=0.2"
            )
            #plt.title("Significant Causal Links → ASP_Quarterly (v(t))", fontsize=14)
            plt.axis("off")
            plt.tight_layout()
            st.pyplot(fig_graph)

    except Exception as e:
        st.warning(f"Failed to draw causal graph: {e}")
else:
    st.info("⬆️ Please upload the causal graph CSV to view the network.")

# -----------------------------------------------------------------------------
# 4. VISUALIZATION 2 – TEMPORAL TRENDS
# -----------------------------------------------------------------------------
st.markdown("## 2️⃣  Causal Coefficient Overview")
lookback = st.slider("Quarters to look back", 3, 60, 5)
df_target = (df_ts[df_ts["date"] <= pd.to_datetime(target_dt)]
             .sort_values("date")
             .tail(lookback))

fig1, ax_base = plt.subplots(figsize=(12, 6))
palette = cm.get_cmap("tab10").colors
axes = [ax_base]
for i in range(1, len(col_target)):
    twin = ax_base.twinx()
    twin.spines["right"].set_position(("axes", 1 + 0.1*(i-1)))
    axes.append(twin)

lines = []
for idx, (ax, col) in enumerate(zip(axes, col_target)):
    color = palette[idx % len(palette)]
    line, = ax.plot(df_target["date"], df_target[col],
                    label=col, color=color, linestyle="--", marker="o")
    ax.set_ylabel(col, color=color)
    ax.tick_params(axis="y", labelcolor=color)
    lines.append(line)

ax_base.set_xlabel("Date")
ax_base.set_xticklabels(df_target["date"].dt.strftime("%Y-%m-%d"),
                        rotation=45)
ax_base.legend(lines, [l.get_label() for l in lines], loc="lower left")
ax_base.grid(axis="x", linestyle=":", alpha=0.5)
fig1.tight_layout()
st.pyplot(fig1)

# -----------------------------------------------------------------------------
# 5. VISUALIZATION 3 – CORRELATION VS CAUSATION
# -----------------------------------------------------------------------------
st.markdown("## 3️⃣  Correlation vs. Causation")

df_q_sub = df_q[df_q["dt"] <= pd.to_datetime(target_dt)]
if df_q_sub.empty:
    st.warning("No quarterly data up to selected date."); st.stop()

y = df_q_sub["v"]
fig2, axes = plt.subplots(1, len(col_target), figsize=(5*len(col_target), 5))
if len(col_target) == 1: axes = [axes]

for i, var in enumerate(col_target):
    x = df_q_sub[[var]]

    # Regression line
    lr = LinearRegression().fit(x, y)
    y_pred = lr.predict(x)

    # Causal line (scaled)
    coef = coef_input[var]
    scaler = MinMaxScaler((y.min(), y.max()))
    causal_scaled = scaler.fit_transform(
        (lr.intercept_ + x * coef).values.reshape(-1,1))

    axes[i].scatter(x, y, s=18, label="Data")
    axes[i].plot(x, y_pred, label="Regression", lw=1.5)
    axes[i].plot(x, causal_scaled,
                 label=f"Causal (coef={coef:.2f})", ls="--")

    axes[i].set_title(var, fontsize=10)
    axes[i].set_xlabel(var, fontsize=9)
    if i == 0: axes[i].set_ylabel("ASP (Quarterly)", fontsize=9)
    axes[i].legend(fontsize=7)
    axes[i].grid(linestyle=":", alpha=0.4)

#fig2.suptitle("Correlation vs. Causation", fontsize=14)
fig2.tight_layout()
st.pyplot(fig2)

# -----------------------------------------------------------------------------
# 6. FOOTER
# -----------------------------------------------------------------------------
st.markdown("---\n💡 **Tip:** change date or Top-N to explore different slices.")
