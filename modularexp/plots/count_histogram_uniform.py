import plotly
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def histogram_from_counts(counts, bins) -> go.Figure:
    names = ["a (uniform)", "b (uniform)", "c (uniform)", "d (uniform)"]
    overall_labels = []
    overall_hist = []
    for i in range(len(counts)):
        full_range = torch.arange(counts[i].shape[0])
        bin_indices = torch.bucketize(full_range, bins[i], right=False)
        hist = torch.zeros((len(bins[i]) + 1), dtype=torch.int64)
        hist = hist.index_add(0, bin_indices, counts[i])
        overall_hist.append(hist)
        full_bins = torch.cat((torch.tensor([0]), bins[i]), 0)  # including left
        # Labels [0, bins[0]), [bins[1], bins[2]) etc
        labels = [f"[{full_bins[idx].round()}, {full_bins[idx + 1].round()})" for idx in range(len(full_bins) - 1)]
        labels.append(f"{full_bins[-1].round()}, ∞)")
        overall_labels.append(labels)
    fig = make_subplots(1, 2)
    col_idx = 0
    for idx in range(len(counts)):
        fig.add_trace(go.Bar(x=overall_labels[idx], y=overall_hist[idx].tolist(), name=names[idx]), row=1, col=col_idx + 1)
        if idx >= 1:
            col_idx = 1

    fig.update_layout(height=650, width=2000, margin={"l": 5, "r": 5, "t": 5, "b": 0},
                      title_text="Parameter distributions", xaxis=dict(title=""),
                      yaxis=dict(title="Count"), font={"size": 35, "family": "Serif"}, legend={"font": {"size": 40}, "orientation": "h", "xanchor": "right", "yanchor": "bottom", "x": 1, "y": 1.1, "entrywidth": 280})
    return fig

checkpoint_path = "./checkpoint_uniform_operands.pth"

checkpoint = torch.load(checkpoint_path)
counts = checkpoint["counts"]

# D
counts_d = counts[3, :102]
bins_d = torch.tensor([19.5, 39.5, 59.5, 79.5], dtype=torch.float) / 100 * counts_d.shape[0]

counts_a = counts[0]
bins_a = torch.tensor([19.999999999, 39.999999999, 59.999999999, 79.999999999], dtype=torch.float) / 100 * counts_a.shape[0]
counts_b = counts[1]
bins_b = torch.tensor([19.999999999, 39.999999999, 59.999999999, 79.999999999], dtype=torch.float) / 100 * counts_b.shape[0]
counts_c = counts[2, :102]
bins_c = torch.tensor([19.5, 39.5, 59.5, 79.5], dtype=torch.float) / 100 * counts_c.shape[0]

fig = histogram_from_counts(counts=[counts_a, counts_b, counts_c, counts_d], bins=[bins_a, bins_b, bins_c, bins_d])
fig.write_image("histogram_uniform.png")
# fig.show()

