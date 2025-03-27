import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

valid1 = pd.read_csv('./modularexp/plots/benford/validbenford1.csv')
valid1 = valid1.rename(columns={"Step": "epoch", "benford - valid_arithmetic_perfect": "accuracy"})
valid2 = pd.read_csv('./modularexp/plots/benford/validbenford2.csv')
valid2 = valid2.rename(columns={"Step": "epoch", "benford2 - valid_arithmetic_perfect": "accuracy"})
test1 = pd.read_csv('./modularexp/plots/benford/testbenford1.csv')
test1 = test1.rename(columns={"Step": "epoch", "benford - test_arithmetic_perfect": "accuracy"})
test2 = pd.read_csv('./modularexp/plots/benford/testbenford2.csv')
test2 = test2.rename(columns={"Step": "epoch", "benford2 - test_arithmetic_perfect": "accuracy"})

valid = pd.concat([valid1, valid2])
test = pd.concat([test1, test2])

fig = go.Figure()

fig.add_trace(go.Scatter(x=valid.epoch, y=valid.accuracy, mode="lines", name="valid"))
fig.add_trace(go.Scatter(x=test.epoch, y=test.accuracy, mode="lines", name="test"))

fig.update_layout(width=800, height=460, font=dict(size=30, family="serif", color="black"),
                  legend=dict(font=dict(size=30, family="serif"), yanchor="top", y=0.6, xanchor="right", x=0.9), xaxis_title="Epoch",
                  yaxis_title="Accuracy")

# fig.show()
fig.write_image("validbenford.png")