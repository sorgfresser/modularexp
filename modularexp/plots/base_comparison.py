import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

base999test = pd.read_csv("./modularexp/plots/base/base999test.csv")
base999test = base999test.rename(columns={"Step": "epoch", "base999 - test_arithmetic_perfect": "accuracy"})

base999valid = pd.read_csv("./modularexp/plots/base/base999valid.csv")
base999valid = base999valid.rename(columns={"Step": "epoch", "base999 - valid_arithmetic_perfect": "accuracy"})

base1000test = pd.read_csv("./modularexp/plots/base/base1000test.csv")
base1000test = base1000test.rename(columns={"Step": "epoch", "uniformbenford - test_arithmetic_perfect": "accuracy"})

base1000valid = pd.read_csv("./modularexp/plots/base/base1000valid.csv")
base1000valid = base1000valid.rename(columns={"Step": "epoch", "uniformbenford - valid_arithmetic_perfect": "accuracy"})

base1013test1 = pd.read_csv("./modularexp/plots/base/base1013test1.csv")
base1013test1 = base1013test1.rename(columns={"Step": "epoch", "base1013 - test_arithmetic_perfect": "accuracy"})
base1013test2 = pd.read_csv("./modularexp/plots/base/base1013test2.csv")
base1013test2 = base1013test2.rename(columns={"Step": "epoch", "base1013 - test_arithmetic_perfect": "accuracy"})
base1013test = pd.concat([base1013test1, base1013test2], ignore_index=True)

base1013valid1 = pd.read_csv("./modularexp/plots/base/base1013valid1.csv")
base1013valid1 = base1013valid1.rename(columns={"Step": "epoch", "base1013 - valid_arithmetic_perfect": "accuracy"})
base1013valid2 = pd.read_csv("./modularexp/plots/base/base1013valid2.csv")
base1013valid2 = base1013valid2.rename(columns={"Step": "epoch", "base1013 - valid_arithmetic_perfect": "accuracy"})
base1013valid = pd.concat([base1013valid1, base1013valid2], ignore_index=True)

base1279test1 = pd.read_csv("./modularexp/plots/base/base1279test1.csv")
base1279test1 = base1279test1.rename(columns={"Step": "epoch", "base1279 - test_arithmetic_perfect": "accuracy"})
base1279test2 = pd.read_csv("./modularexp/plots/base/base1279test2.csv")
base1279test2 = base1279test2.rename(columns={"Step": "epoch", "base1279 - test_arithmetic_perfect": "accuracy"})
base1279test = pd.concat([base1279test1, base1279test2], ignore_index=True)

base1279valid1 = pd.read_csv("./modularexp/plots/base/base1279valid1.csv")
base1279valid1 = base1279valid1.rename(columns={"Step": "epoch", "base1279 - valid_arithmetic_perfect": "accuracy"})
base1279valid2 = pd.read_csv("./modularexp/plots/base/base1279valid2.csv")
base1279valid2 = base1279valid2.rename(columns={"Step": "epoch", "base1279 - valid_arithmetic_perfect": "accuracy"})
base1279valid = pd.concat([base1279valid1, base1279valid2], ignore_index=True)

base999test = base999test.iloc[:1000]
base1000test = base1000test.iloc[:1000]
base1013test = base1013test.iloc[:1000]
base1279test = base1279test.iloc[:1000]
base999valid = base999valid.iloc[:1000]
base1000valid = base1000valid.iloc[:1000]
base1013valid = base1013valid.iloc[:1000]
base1279valid = base1279valid.iloc[:1000]

base999test["run"] = "base999"
base1000test["run"] = "base1000"
base1013test["run"] = "base1013"
base1279test["run"] = "base1279"
base999valid["run"] = "base999"
base1000valid["run"] = "base1000"
base1013valid["run"] = "base1013"
base1279valid["run"] = "base1279"

df_test = pd.concat([base999test, base1000test, base1013test, base1279test], ignore_index=True)
df_valid = pd.concat([base999valid, base1000valid, base1013valid, base1279valid], ignore_index=True)

colors = {
    "base999": "#AB63FA",
    "base1000": "red",
    "base1013": "#00CC96",
    "base1279": "orange"
}

fig = make_subplots(1, 2, shared_yaxes=True, shared_xaxes=True, subplot_titles=["Test", "Validation"])
for run in df_test["run"].unique():
    # Filter data for current run
    df_test_run = df_test[df_test["run"] == run]
    df_valid_run = df_valid[df_valid["run"] == run]

    # Add test trace in the first subplot (row=1, col=1)
    fig.add_trace(
        go.Scatter(x=df_test_run["epoch"], y=df_test_run["accuracy"], mode="lines", name=run, legendgroup=run,
                   showlegend=True, line=dict(color=colors[run])),
        row=1, col=1
    )

    # Add validation trace in the second subplot (row=2, col=1)
    fig.add_trace(
        go.Scatter(x=df_valid_run["epoch"], y=df_valid_run["accuracy"], mode="lines", name=run, legendgroup=run,
                   showlegend=False, line=dict(color=colors[run])),
        row=1, col=2
    )
# fig.add_trace(df_valid, row=1, col=2)
# fig_test = px.line(df_test, x="epoch", y="accuracy", color="run",
#                      title="Test Accuracy vs Epoch")
# fig_test.show()
#
# fig_valid = px.line(df_valid, x="epoch", y="accuracy", color="run",
#                     title="Validation Accuracy vs Epoch")
# fig_valid.show()
# fig.show()
fig.update_layout(
    width=1000,
    height=400,
    font={"size": 22, "family": "serif", "color": "black"},
    legend={"font": {"size": 22, "color": "black", "family": "serif"}, "itemwidth": 30},
    yaxis_title="Accuracy",

)
fig.update_annotations(font=dict(size=22, color="black", family="serif"))

fig.update_xaxes(title_text="Epoch", row=1, col=1)
fig.update_xaxes(title_text="Epoch", row=1, col=2)
fig.write_image("base_comparison.png")
