import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

valid23 = pd.read_csv("./modularexp/plots/valid/valid23.csv")
valid23 = valid23.rename(columns={"Step": "epoch", "benford2 - valid_arithmetic_acc_23_2": "accuracy"})

valid46 = pd.read_csv("./modularexp/plots/valid/valid46.csv")
valid46 = valid46.rename(columns={"Step": "epoch", "benford2 - valid_arithmetic_acc_46_2": "accuracy"})

valid69 = pd.read_csv("./modularexp/plots/valid/valid69.csv")
valid69 = valid69.rename(columns={"Step": "epoch", "benford2 - valid_arithmetic_acc_69_2": "accuracy"})

valid92 = pd.read_csv("./modularexp/plots/valid/valid92.csv")
valid92 = valid92.rename(columns={"Step": "epoch", "benford2 - valid_arithmetic_acc_92_2": "accuracy"})

valid_overall = pd.read_csv("./modularexp/plots/valid/valid_overall.csv")
valid_overall = valid_overall.rename(columns={"Step": "epoch", "benford2 - valid_arithmetic_perfect": "accuracy"})

fig = go.Figure()

fig.add_trace(go.Scatter(x=valid23.epoch, y=valid23.accuracy, mode="lines", name="Mod 23"))
fig.add_trace(go.Scatter(x=valid46.epoch, y=valid46.accuracy, mode="lines", name="Mod 46"))
fig.add_trace(go.Scatter(x=valid69.epoch, y=valid69.accuracy, mode="lines", name="Mod 69"))
fig.add_trace(go.Scatter(x=valid92.epoch, y=valid92.accuracy, mode="lines", name="Mod 92"))
fig.add_trace(go.Scatter(x=valid_overall.epoch, y=valid_overall.accuracy, mode="lines", name="Overall"))

fig.update_layout(width=800, height=500, font=dict(size=25, family="serif", color="black"),
                  legend={"font": {"size": 25, "family": "serif", "color": "black"}, "xanchor": "right",
                          "yanchor": "top", "x":0.91, "y": 0.7}, xaxis_title="Epochs",
                  yaxis_title="Accuracy", )

# fig.show()
fig.write_image("validcomparison.png")