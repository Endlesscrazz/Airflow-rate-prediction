import matplotlib.pyplot as plt
import numpy as np

# Define your MLP architecture
# For hidden layers with many neurons, we'll draw a representative number to keep the diagram clean.
# Actual neuron counts will be labeled.
input_neurons = 5
hidden1_neurons_actual = 20
hidden2_neurons_actual = 10
hidden3_neurons_actual = 5
output_neurons = 1

# Number of neurons to *draw* for visualization purposes (can be less than actual for large layers)
hidden1_neurons_draw = 7 # Representative
hidden2_neurons_draw = 5 # Representative
hidden3_neurons_draw = 3 # Representative

layer_sizes_to_draw = [input_neurons, hidden1_neurons_draw, hidden2_neurons_draw, hidden3_neurons_draw, output_neurons]
actual_hidden_layer_neuron_counts = [hidden1_neurons_actual, hidden2_neurons_actual, hidden3_neurons_actual]


# Input feature labels (abbreviated)
input_feature_labels = [
    'ΔT_log', 
    'Area_log', 
    'Rate_init_norm', 
    'Std_ΔT_overall', 
    'T_max_init'
]
output_label = ['Airflow Rate']

# Create the figure
fig, ax = plt.subplots(1, 1, figsize=(10, 7)) # Adjust figsize as needed
ax.axis('off') # Turn off the axis lines and ticks

# Adjust left, right, bottom, top to control spacing and layout
left_margin = 0.15 
right_margin = 0.85 
bottom_margin = 0.1
top_margin = 0.9

# Modify draw_neural_net to accept actual neuron counts for labeling
def draw_custom_neural_net(ax, left, right, bottom, top, layer_sizes_draw, actual_hidden_counts, input_labels=None, output_labels=None):
    n_layers = len(layer_sizes_draw)
    # Calculate max neurons to draw for vertical spacing
    max_drawn_neurons = 0
    if layer_sizes_draw: 
        max_drawn_neurons = float(max(layer_sizes_draw))
    
    if max_drawn_neurons == 0: 
        v_spacing = 0
    else:
        v_spacing = (top - bottom) / max_drawn_neurons

    if n_layers <=1: 
        h_spacing = 0
    else:
        h_spacing = (right - left) / float(n_layers - 1)
    
    node_radius = 0.04 
    edge_alpha = 0.3 
    input_label_offset_multiplier = 2.8 
    output_label_offset_multiplier = 3.0 

    # Nodes
    for n, layer_size_draw in enumerate(layer_sizes_draw):
        current_layer_height = v_spacing * (layer_size_draw -1)
        layer_top = current_layer_height / 2. + (top + bottom) / 2.

        for m in range(layer_size_draw):
            x = left + n * h_spacing
            y = layer_top - m * v_spacing
            circle = plt.Circle((x, y), node_radius, color='skyblue', ec='black', zorder=4)
            ax.add_artist(circle)
            
            # Add labels
            if n == 0 and input_labels and m < len(input_labels): # Input layer
                ax.text(x - node_radius * input_label_offset_multiplier, y, input_labels[m], 
                        ha='right', va='center', fontsize=12) # Increased fontsize
            elif n == n_layers - 1 and output_labels and m < len(output_labels): # Output layer
                 ax.text(x + node_radius * output_label_offset_multiplier, y, output_labels[m], 
                         ha='left', va='center', fontsize=14) # Increased fontsize
            elif 0 < n < n_layers -1 : # Hidden Layers
                actual_count_idx = n - 1 
                if m == 0 and actual_count_idx < len(actual_hidden_counts): 
                    label_y_pos = (v_spacing * (layer_size_draw -1) / 2. + (top + bottom) / 2.) + v_spacing * 0.7 # Adjusted y offset for label
                    ax.text(x, label_y_pos, 
                            f'{actual_hidden_counts[actual_count_idx]} neurons\n(tanh)', 
                            ha='center', va='bottom', fontsize=10) # Increased fontsize

    # Edges
    for n, (layer_size_a_draw, layer_size_b_draw) in enumerate(zip(layer_sizes_draw[:-1], layer_sizes_draw[1:])):
        current_layer_a_height = v_spacing * (layer_size_a_draw - 1)
        layer_top_a = current_layer_a_height / 2. + (top + bottom) / 2.
        
        current_layer_b_height = v_spacing * (layer_size_b_draw - 1)
        layer_top_b = current_layer_b_height / 2. + (top + bottom) / 2.
        
        for m_a in range(layer_size_a_draw):
            for m_b in range(layer_size_b_draw):
                x_a = left + n * h_spacing
                y_a = layer_top_a - m_a * v_spacing
                x_b = left + (n + 1) * h_spacing
                y_b = layer_top_b - m_b * v_spacing
                line = plt.Line2D([x_a, x_b], [y_a, y_b], c='black', lw=0.3, alpha=edge_alpha, zorder=1)
                ax.add_artist(line)

# Call the drawing function
draw_custom_neural_net(ax, left_margin, right_margin, bottom_margin, top_margin, 
                       layer_sizes_to_draw, actual_hidden_layer_neuron_counts,
                       input_feature_labels, output_label)

plt.title("MLP Architecture for Airflow Prediction\n(5 Inputs -> 20 tanh -> 10 tanh -> 5 tanh -> 1 Output)", fontsize=15, y=1.02) # Increased title fontsize and adjusted y
plt.savefig("mlp_architecture_diagram_larger_font.png", dpi=300, bbox_inches='tight')
print("MLP architecture diagram saved as mlp_architecture_diagram_larger_font.png")
plt.show() # Uncomment to display if running locally
