# functions/plotting.py
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np

def plot_bar_chart(data, x, y, title=None):
    fig, ax = plt.subplots()
    sns.barplot(data=data, x=x, y=y, ax=ax)
    if title:
        ax.set_title(title)
    st.pyplot(fig)

# Add more plotting functions as needed

def plot_histogram(data, column, bins=20, title=None):
    fig, ax = plt.subplots()
    ax.hist(data[column], bins=bins)
    if title:
        ax.set_title(title)
    st.pyplot(fig)

# ...add other plotting functions as needed

def plot_protein_counts_bar(protein_counts, top_n):
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
    sns.barplot(y=protein_counts.index, x=protein_counts.values, ax=ax, orient='h')
    ax.set_ylabel("Protein Name")
    ax.set_xlabel("Number of Activity Points")
    ax.set_title(f"Number of Activity Points per Protein (Top {top_n})")
    st.pyplot(fig)

def plot_bioactivity_type_pie_hist(bioactivity_counts, percent, selected_group, df_hist_numeric, log_scale):
    # Pie chart
    values = np.array(bioactivity_counts.tolist(), dtype=float)
    labels = [str(label) for label in list(bioactivity_counts.index)]
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.set_title("Bioactivity Type Distribution")
    st.pyplot(fig)
    # Histogram for selected group
    if isinstance(df_hist_numeric, type(None)) or df_hist_numeric.empty:
        st.info("No numeric values available for this group.")
        return
    values_array = df_hist_numeric["Value"].to_numpy(dtype=float)
    lower, upper = np.percentile(values_array, 1), np.percentile(values_array, 99)
    clipped = values_array[(values_array >= lower) & (values_array <= upper)]
    if len(clipped) < len(values_array):
        st.caption(f"Showing values between {lower:.2g} and {upper:.2g} (1st-99th percentile). {len(values_array)-len(clipped)} outliers clipped.")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    if log_scale:
        clipped = clipped[clipped > 0]
        if len(clipped) == 0:
            st.info("No positive values to display on log scale.")
            return
        bins = np.logspace(np.log10(clipped.min()), np.log10(clipped.max()), 10)
        sns.histplot(clipped, bins=bins, ax=ax2, kde=True)
        ax2.set_xscale('log')
        ax2.set_xlabel("Bioactivity Value (log scale)")
    else:
        sns.histplot(clipped, bins=10, ax=ax2, kde=True)
        ax2.set_xlim((float(lower), float(upper)))
        ax2.set_xlabel("Bioactivity Value")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Distribution of Bioactivity Values for {selected_group}")
    st.pyplot(fig2)

def plot_categorical_pie_bar(value_counts, selected_col):
    values = np.array(value_counts.tolist(), dtype=float)
    labels = [str(label) for label in list(value_counts.index)]
    col1, col2 = st.columns([1, 2])
    with col1:
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.set_title(f"{selected_col} Distribution (Pie)")
        st.pyplot(fig)
    with col2:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax2)
        ax2.set_xlabel(selected_col)
        ax2.set_ylabel("Count")
        ax2.set_title(f"{selected_col} Distribution (Barplot)")
        plt.xticks(rotation=45)
        st.pyplot(fig2)

def plot_numeric_histogram(col_data_numeric, selected_col, log_scale):
    values_array = col_data_numeric.to_numpy(dtype=float)
    lower, upper = np.percentile(values_array, 1), np.percentile(values_array, 99)
    clipped = values_array[(values_array >= lower) & (values_array <= upper)]
    if len(clipped) < len(values_array):
        st.caption(f"Showing values between {lower:.2g} and {upper:.2g} (1st-99th percentile). {len(values_array)-len(clipped)} outliers clipped.")
    fig, ax = plt.subplots(figsize=(6, 4))
    if log_scale:
        clipped = clipped[clipped > 0]
        if len(clipped) == 0:
            st.info("No positive values to display on log scale.")
            return
        bins = np.logspace(np.log10(clipped.min()), np.log10(clipped.max()), 10)
        sns.histplot(clipped, bins=bins, ax=ax, kde=True)
        ax.set_xscale('log')
        ax.set_xlabel(f"{selected_col} (log scale)")
    else:
        sns.histplot(clipped, bins=10, ax=ax, kde=True)
        ax.set_xlim((float(lower), float(upper)))
        ax.set_xlabel(selected_col)
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of {selected_col}")
    st.pyplot(fig)

def plot_interactive_networkx_pyvis(G, title="Network Visualization", test_mode=False):
    from pyvis.network import Network
    import streamlit as st
    from streamlit.components.v1 import html
    # 1. Print network size
    st.write(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    # 2. Minimal test network if requested
    if test_mode:
        import networkx as nx
        G = nx.Graph()
        G.add_edge('A', 'B')
        G.add_edge('B', 'C')
        G.add_edge('C', 'A')
        st.info("Showing minimal test network (triangle)")
    # Use default background and node/edge colors
    net = Network(height="600px", width="100%", notebook=False)
    
    # Manually add nodes and edges to ensure labels are plotted
    for node in G.nodes():
        net.add_node(node, label=str(node)) # Ensure label is set
        
    for source, target, attr in G.edges(data=True):
        net.add_edge(source, target, value=attr.get('combined_score', 1))

    net.barnes_hut()
    net.show_buttons(filter_=['physics'])
    html_str = net.generate_html()
    st.subheader(title)
    html(html_str, height=650, width=900, scrolling=True)

def plot_static_networkx(G, title="Static Network Visualization"):
    import matplotlib.pyplot as plt
    import streamlit as st
    import networkx as nx
    import numpy as np
    from matplotlib.colors import Normalize
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    # Node coloring by degree
    degrees = dict(G.degree())
    node_color_vals = [float(degrees[n]) for n in G.nodes()]
    cmap = plt.get_cmap('viridis')
    # Edge thickness by score (weight)
    edge_weights = [G[u][v].get('weight', 1.0) for u, v in G.edges()]
    # Normalize edge weights for width (e.g., 1-8)
    if edge_weights:
        min_w, max_w = min(edge_weights), max(edge_weights)
        if max_w > min_w:
            widths = [1 + 7 * (w - min_w) / (max_w - min_w) for w in edge_weights]
        else:
            widths = [2.0 for _ in edge_weights]
    else:
        widths = []
    vmin = float(min(node_color_vals)) if len(node_color_vals) > 0 else 0.0
    vmax = float(max(node_color_vals)) if len(node_color_vals) > 0 else 1.0
    nodes = nx.draw_networkx_nodes(G, pos, node_color=node_color_vals, cmap=cmap, node_size=500, vmin=vmin, vmax=vmax, ax=ax)
    edges = nx.draw_networkx_edges(G, pos, edge_color='gray', width=widths, ax=ax)
    nx.draw_networkx_labels(G, pos, font_color='black', ax=ax)
    ax.set_title(title)
    # Add colorbar for node degree
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7)
    cbar.set_label('Node Degree')
    st.pyplot(fig)
    plt.close(fig) 