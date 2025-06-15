import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
import datetime
from PIL import Image

# --------------------------------------------------  PAGE HEADER
st.set_page_config(page_title="SKT DRAM Price Causality Visualization Dashboard",
                   layout="wide")

col_logo, _ = st.columns([2, 8])
with col_logo:
    st.image("impactiveAI_logo.png", width=300)

st.markdown(
    """
    <h2 style='text-align:center;margin-top:10px;margin-bottom:10px'>
        Causality Visualization Dashboard<br>
        <span style='font-size:24px;'>SKT DRAM Price</span>
    </h2>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------  1. DATA UPLOAD
st.markdown("### 📂 Upload Files")
u1, u2 = st.columns(2)

with u1:
    st.markdown(
        "<div style='margin-bottom:-50px;'><label style='font-size:18px;'>1. Quarterly ASP</label></div>",
        unsafe_allow_html=True,
    )
    q_file = st.file_uploader("", type=["csv"], key="q")

with u2:
    st.markdown(
        "<div style='margin-bottom:-50px;'><label style='font-size:18px;'>2. Causal Graph</label></div>",
        unsafe_allow_html=True,
    )
    graph_file = st.file_uploader("", type=["csv"], key="g")

if (q_file is None) or (graph_file is None):
    st.info("⬆️  Upload both CSV files to proceed.")
    st.stop()

# --------------------------------------------------  2. LOAD DATA
@st.cache_data(show_spinner="Loading quarterly…")
def load_quarterly(csv):
    df = pd.read_csv(csv)
    df["dt"] = pd.to_datetime(df["dt"])
    return df

df_q = load_quarterly(q_file)

graph_db = pd.read_csv(graph_file)
graph_db["dt"] = pd.to_datetime(graph_db["dt"])

# --------------------------------------------------  3. BUILD AGGREGATED RESULT (df_ts)
src_cols = [c for c in df_q.columns if c not in {"dt", "grain_id", "v", "ASP_monthly"}]
df_causal = pd.DataFrame({"source": src_cols})

for dt, g in graph_db.groupby("dt"):
    tmp = g[["source", "Causal_effect"]]
    idx = tmp.groupby("source")["Causal_effect"].apply(lambda x: x.abs().idxmax())
    g_max = (
        tmp.loc[idx]
        .reset_index(drop=True)
        .rename(columns={"Causal_effect": dt})
    )
    df_causal = df_causal.merge(g_max, on="source", how="left")

df_causal.fillna(0, inplace=True)
df_ts = (
    df_causal.set_index("source")
    .T.reset_index()
    .rename(columns={"index": "date"})
)
df_ts["date"] = pd.to_datetime(df_ts["date"])

# --------------------------------------------------  4. TARGET ADJUSTMENT
st.markdown("### 🎯 Target Adjustment")
ad1, ad2 = st.columns(2)

min_date, max_date = df_ts["date"].min(), df_ts["date"].max()

with ad1:
    target_dt = st.date_input(
        "📅 Select target date",
        value=datetime.date(2022, 10, 3),
        min_value=min_date.date(),
        max_value=max_date.date(),
    )
with ad2:
    n_vars = st.slider("🔢 Top-N variables (by |value|)", 1, 10, 3)

latest = (
    df_ts[df_ts["date"] == pd.to_datetime(target_dt)]
    .drop(columns="date")
    .T.reset_index()
)
latest.columns = ["variable", "value"]
latest["abs"] = latest["value"].abs()

col_target_df = (
    latest.sort_values("abs", ascending=False).head(n_vars).reset_index(drop=True)
)
col_target = col_target_df["variable"].tolist()
coef_input = dict(zip(col_target_df["variable"], col_target_df["value"]))

st.markdown("### 💡 Key Features with Strongest Causal Impact")
st.dataframe(col_target_df[["variable", "value"]], use_container_width=True)

# --------------------------------------------------  5. CAUSAL KNOWLEDGE GRAPH
st.markdown("## 1️⃣ Causal Knowledge Graph")
causal_table = graph_db[graph_db["dt"] == pd.to_datetime(target_dt)]

if causal_table.empty:
    st.info("No causal links for the selected date.")
else:
    G = nx.DiGraph()
    colors, widths, spring = [], [], {}
    for _, row in causal_table.iterrows():
        s = f"{row['source']}(t-{row['lag']})"
        t = f"{row['target']}(t)"
        eff, lag = row["Causal_effect"], row["lag"]
        G.add_edge(s, t)
        colors.append("blue" if eff > 0 else "red")
        widths.append(abs(eff) * 2)
        spring[(s, t)] = 1 / lag
    nx.set_edge_attributes(G, spring, "weight")
    pos = nx.spring_layout(G, seed=42, weight="weight")

        # Prepare per-node style arrays
    node_colors        = []
    node_sizes         = []
    node_border_colors = []
    node_border_lw     = []

    for node in G.nodes():
        # strip anything after first "(" so it works for both "(t-…)" and "(t)"
        base_name = node.split('(')[0]

        if base_name in col_target:
            node_colors.append('#ffcc33')   # gold-ish
            node_sizes.append(2000)         # bigger node
            node_border_colors.append('grey')
            node_border_lw.append(1.5)      # thicker outline
        else:
            node_colors.append('lightblue')
            node_sizes.append(1000)
            node_border_colors.append('#888888')  # subtle grey border
            node_border_lw.append(0.8)

    fig_graph = plt.figure(figsize=(9, 6))
    nx.draw(
        G, pos,
        with_labels=True,
        node_color=node_colors,
        node_size=node_sizes,
        edge_color=colors,
        width=widths,
        edgecolors=node_border_colors,      # outline colours
        linewidths=node_border_lw,          # outline widths
        font_size=9,
        arrowsize=12,
        connectionstyle='arc3,rad=0.2'
    )
    plt.axis("off")
    plt.tight_layout()
    st.pyplot(fig_graph)

# --------------------------------------------------  6. TEMPORAL TRENDS
st.markdown("## 2️⃣ Causal Coefficient Overview")
lookback = st.slider("📈 Quarters to look back", 3, 60, 5)

df_hist = (
    df_ts[df_ts["date"] <= pd.to_datetime(target_dt)]
    .sort_values("date")
    .tail(lookback)
)
fig1, ax_base = plt.subplots(figsize=(12, 6))
palette = cm.get_cmap("tab10").colors
axes = [ax_base]
for i in range(1, len(col_target)):
    twin = ax_base.twinx()
    twin.spines["right"].set_position(("axes", 1 + 0.1 * (i - 1)))
    axes.append(twin)

plot_lines = []
for idx, (ax_i, col) in enumerate(zip(axes, col_target)):
    c = palette[idx % len(palette)]
    ln, = ax_i.plot(
        df_hist["date"], df_hist[col], marker="o", ls="--", color=c, label=col
    )
    ax_i.set_ylabel(col, color=c)
    ax_i.tick_params(axis="y", labelcolor=c)
    plot_lines.append(ln)

ax_base.set_xlabel("Date")
ax_base.set_xticklabels(df_hist["date"].dt.strftime("%Y-%m-%d"), rotation=45)
ax_base.legend(plot_lines, [l.get_label() for l in plot_lines], loc="lower left")
ax_base.grid(axis="x", ls=":", alpha=0.5)
st.pyplot(fig1)

# --------------------------------------------------  7. CORRELATION VS CAUSATION
st.markdown("## 3️⃣ Correlation vs. Causation")
df_q_sub = df_q[df_q["dt"] <= pd.to_datetime(target_dt)]
if df_q_sub.empty:
    st.warning("No quarterly data before selected date.")
    st.stop()

y = df_q_sub["v"]
fig2, axes = plt.subplots(1, len(col_target), figsize=(5 * len(col_target), 5))
if len(col_target) == 1:
    axes = [axes]

for i, var in enumerate(col_target):
    x = df_q_sub[[var]]
    lr = LinearRegression().fit(x, y)
    y_pred = lr.predict(x)
    coef = coef_input[var]
    scaler = MinMaxScaler((y.min(), y.max()))
    causal_scaled = scaler.fit_transform((lr.intercept_ + x * coef).values.reshape(-1, 1))

    axes[i].scatter(x, y, s=18, label="Data")
    axes[i].plot(x, y_pred, lw=1.5, label="Regression")
    axes[i].plot(x, causal_scaled, ls="--", label=f"Causal (coef={coef:.2f})")
    axes[i].set_title(var)
    if i == 0:
        axes[i].set_ylabel("ASP (Quarterly)")
    axes[i].legend(fontsize=7)
    axes[i].grid(ls=":", alpha=0.4)

st.pyplot(fig2)

# --------------------------------------------------  FOOTER
st.markdown("---\n💡 **Tip:** change date or Top-N variables to explore different slices.")