import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

# Your results data
levels = ['Level 1<br>(7 classes)', 'Level 2<br>(56 classes)',
          'Level 3<br>(154 classes)', 'Level 4<br>(532 classes)']
level_names_simple = ['Level 1', 'Level 2', 'Level 3', 'Level 4']

# Accuracy data for each layer
layer_0 = [0.386, 0.151, 0.098, 0.028]
layer_1 = [0.544, 0.381, 0.367, 0.420]
layer_2 = [0.585, 0.479, 0.460, 0.559]
layer_3 = [0.663, 0.627, 0.628, 0.691]
layer_4 = [0.779, 0.765, 0.751, 0.844]
layer_5 = [0.783, 0.800, 0.797, 0.898]  # BEST
layer_6 = [0.717, 0.638, 0.646, 0.665]

layers = np.arange(7)

# F1 scores for Layer 5
layer_5_acc = [0.783, 0.800, 0.797, 0.898]
layer_5_f1 = [0.782, 0.796, 0.780, 0.878]

# ============================================================
# GRAPH 1: Line plot - Accuracy across all layers for each EC level
# ============================================================

fig1 = go.Figure()

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Add traces for each EC level
for i, (name, color) in enumerate(zip(level_names_simple, colors)):
    accuracies = [layer_0[i], layer_1[i], layer_2[i], layer_3[i],
                  layer_4[i], layer_5[i], layer_6[i]]

    fig1.add_trace(go.Scatter(
        x=layers,
        y=accuracies,
        mode='lines+markers',
        name=name,
        line=dict(color=color, width=3),
        marker=dict(size=10, color=color),
        hovertemplate='<b>%{fullData.name}</b><br>' +
                      'Layer: %{x}<br>' +
                      'Accuracy: %{y:.3f}<br>' +
                      '<extra></extra>'
    ))

# Add vertical line at Layer 5 (peak)
fig1.add_vline(x=5, line_dash="dash", line_color="red",
               opacity=0.5, line_width=2,
               annotation_text="Layer 5 (Peak)",
               annotation_position="top right",
               annotation_font_size=12,
               annotation_font_color="red")

fig1.update_layout(
    title={
        'text': 'EC Prediction Accuracy Across ESM-2 8M Layers',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 20, 'family': 'Arial Black'}
    },
    xaxis_title='Layer',
    yaxis_title='Accuracy',
    font=dict(size=14),
    hovermode='closest',
    template='plotly_white',
    width=900,
    height=600,
    legend=dict(
        yanchor="bottom",
        y=0.02,
        xanchor="right",
        x=0.98,
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="Black",
        borderwidth=1
    ),
    xaxis=dict(
        tickmode='linear',
        tick0=0,
        dtick=1,
        range=[-0.2, 6.2]
    ),
    yaxis=dict(
        range=[0, 1],
        tickformat='.0%'
    )
)

# Show and save
fig1.show()
fig1.write_html("ec_accuracy_across_layers.html")
print("Graph 1 saved as: ec_accuracy_across_layers.html")

# ============================================================
# GRAPH 2: Grouped bar chart - Layer 5 Performance by EC Level
# ============================================================

fig2 = go.Figure()

# Create x positions for grouped bars
x = np.arange(len(levels))

# Add Accuracy bars
fig2.add_trace(go.Bar(
    x=levels,
    y=layer_5_acc,
    name='Accuracy',
    marker_color='#2ca02c',
    opacity=0.8,
    text=[f'{val:.1%}' for val in layer_5_acc],
    textposition='outside',
    textfont=dict(size=12, color='black'),
    hovertemplate='<b>Accuracy</b><br>' +
                  '%{x}<br>' +
                  'Score: %{y:.3f}<br>' +
                  '<extra></extra>'
))

# Add F1 Score bars
fig2.add_trace(go.Bar(
    x=levels,
    y=layer_5_f1,
    name='F1 Score',
    marker_color='#ff7f0e',
    opacity=0.8,
    text=[f'{val:.1%}' for val in layer_5_f1],
    textposition='outside',
    textfont=dict(size=12, color='black'),
    hovertemplate='<b>F1 Score</b><br>' +
                  '%{x}<br>' +
                  'Score: %{y:.3f}<br>' +
                  '<extra></extra>'
))

fig2.update_layout(
    title={
        'text': 'Layer 5 Performance by EC Hierarchy Level',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 20, 'family': 'Arial Black'}
    },
    xaxis_title='EC Hierarchy Level',
    yaxis_title='Score',
    font=dict(size=14),
    barmode='group',
    template='plotly_white',
    width=900,
    height=600,
    legend=dict(
        yanchor="top",
        y=0.98,
        xanchor="right",
        x=0.98,
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor="Black",
        borderwidth=1
    ),
    yaxis=dict(
        range=[0, 1.05],
        tickformat='.0%'
    ),
    hovermode='x unified'
)

# Show and save
fig2.show()
fig2.write_html("layer5_performance_by_ec_level.html")
print("Graph 2 saved as: layer5_performance_by_ec_level.html")

print("\n✅ Both graphs created successfully!")
print("   - ec_accuracy_across_layers.html")
print("   - layer5_performance_by_ec_level.html")