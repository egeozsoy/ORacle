import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
import json


# Load your data
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


# Update the network graph for each frame of the animation
def update_graph(num, data, graph, ax, pos):
    ax.clear()
    timepoint = data[num][0]
    actor, action, obj = data[num][1]

    if 'stopped' in action:
        if graph.has_edge(actor, obj):
            graph.remove_edge(actor, obj)
    else:
        if not graph.has_edge(actor, obj):
            graph.add_edge(actor, obj, label=action)

    # Draw the graph with a consistent layout and style
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color='skyblue', node_size=500)
    nx.draw_networkx_edges(graph, pos, ax=ax, arrowstyle='->', arrowsize=20)
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=10)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=nx.get_edge_attributes(graph, 'label'), ax=ax, font_size=8)

    ax.set_title(f"Time: {timepoint}", fontsize=15)
    ax.axis('off')


# Main function to create the animation
def create_animation(file_path):
    data = load_data(file_path)

    # Determine all unique nodes from the data
    all_nodes = set()
    for _, (actor, _, obj) in data:
        all_nodes.add(actor)
        all_nodes.add(obj)

    # Create a static layout for the nodes
    static_pos = nx.spring_layout(all_nodes)

    # Set up the plot for the animation
    fig, ax = plt.subplots(figsize=(12, 10))
    graph = nx.DiGraph()
    graph.add_nodes_from(all_nodes)

    # Create the animation
    ani = animation.FuncAnimation(fig, update_graph, frames=len(data), fargs=(data, graph, ax, static_pos), interval=500, repeat=False)

    # Save the animation
    ani.save('surgery_animation.mp4', writer='ffmpeg')


if __name__ == "__main__":
    file_path = 'data/llava_samples/surgery_sg.json'  # Replace with the path to your JSON file
    create_animation(file_path)
