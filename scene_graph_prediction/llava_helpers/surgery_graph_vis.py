from collections import defaultdict

import plotly.graph_objs as go
import networkx as nx
import json

from scene_graph_prediction.llava_helpers.scene_graph_converters import collapse_sgs

FOCUS_ENTITY = 'patient'
SCALE_FACTOR = 5


def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def create_interactive_graph(data):
    timepoints = sorted(set([interaction[0] for interaction in data]))
    G = nx.DiGraph()

    node_positions = {}
    prev_time_point = None
    x_y_orders = [(0, 5), (0, -5), (-1, 5), (-1, -5), (1, 5), (1, -5)]
    # scale by 10
    x_y_orders = [(x * 10, y * 10) for x, y in x_y_orders]
    prev_focus_pred = set()

    for timepoint in timepoints:
        # get sgs till this timepoint
        timepoint_sgs = [elem for elem in data if elem[0] == timepoint]
        curr_focus_pred = set()
        FOCUS_ENTITY_FOUND = False
        for _, (sub, pred, obj) in timepoint_sgs:
            if sub == FOCUS_ENTITY or obj == FOCUS_ENTITY and not pred.startswith('not '):
                FOCUS_ENTITY_FOUND = True
                curr_focus_pred.add((sub, obj, pred))
        if curr_focus_pred == prev_focus_pred:
            continue
        if not FOCUS_ENTITY_FOUND:
            continue
        prev_focus_pred = curr_focus_pred

        # now that we are here, we should add a FOCUS ENTITY node
        focus_node = f'{FOCUS_ENTITY}_{timepoint}'
        if focus_node not in node_positions:
            G.add_node(focus_node, label=FOCUS_ENTITY, size=40)
            node_positions[focus_node] = (timepoint * SCALE_FACTOR, 0)
            if prev_time_point is not None:  # temporal link to previous focus node
                G.add_edge(f'{FOCUS_ENTITY}_{prev_time_point}', focus_node, label='temporal', color='blue', line=dict(dash='dash'))
        prev_time_point = timepoint
        # simply visualize everything that is present in the sub_obj_to_pred where the subject or object is the FOCUS ENTITY
        timepoint_counter = 0
        for (sub, obj, pred) in curr_focus_pred:
            if sub == FOCUS_ENTITY or obj == FOCUS_ENTITY:
                sub_node = f"{sub}_{timepoint}"
                obj_node = f"{obj}_{timepoint}"
                if sub_node not in node_positions:
                    G.add_node(sub_node, label=sub, size=30)
                    node_positions[sub_node] = (timepoint * SCALE_FACTOR + x_y_orders[timepoint_counter][0], x_y_orders[timepoint_counter][1])
                    timepoint_counter += 1
                if obj_node not in node_positions:
                    G.add_node(obj_node, label=obj, size=30)
                    node_positions[obj_node] = (timepoint * SCALE_FACTOR + x_y_orders[timepoint_counter][0], x_y_orders[timepoint_counter][1])
                    timepoint_counter += 1
                G.add_edge(sub_node, obj_node, label=pred, color='red')

    # Initialize separate edge traces for different colors
    temporal_edge_trace = go.Scatter(
        x=[], y=[], line=dict(width=2, dash='dash'), hoverinfo='none', mode='lines', name='Temporal Edge'
    )
    interaction_edge_trace = go.Scatter(
        x=[], y=[], line=dict(width=2), hoverinfo='none', mode='lines', name='Interaction Edge'
    )
    annotations = []
    # Inside the loop where you create edge traces
    for edge in G.edges(data=True):
        x0, y0 = node_positions[edge[0]]
        x1, y1 = node_positions[edge[1]]
        # Distinguish between temporal and interaction edges
        if 'temporal' in edge[2].get('label', ''):
            temporal_edge_trace['x'] += (x0, x1, None)
            temporal_edge_trace['y'] += (y0, y1, None)
            temporal_edge_trace['line']['dash'] = 'dash'  # Set the dash style for temporal edges
        else:
            interaction_edge_trace['x'] += (x0, x1, None)
            interaction_edge_trace['y'] += (y0, y1, None)
            annotations.append(
                dict(
                    ax=x0, ay=y0, axref='x', ayref='y',
                    x=x1, y=y1, xref='x', yref='y',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor='red'
                )
            )

        # Add an annotation for the edge label
        edge_label = edge[2].get('label', '')
        if edge_label:
            x_mid = (x0 + x1) / 2
            y_mid = (y0 + y1) / 2
            annotations.append(
                dict(
                    text=edge_label,
                    x=x_mid,
                    y=y_mid,
                    showarrow=False,
                    font=dict(size=1 if 'temporal' in edge[2].get('label', '') else 10),  # Adjust the font size as needed
                )
            )

    # Now edge_trace['line']['color'] should be a tuple with color values for each line segment.

    node_trace = go.Scatter(
        x=[], y=[], text=[], mode='markers+text', textposition='bottom center', hoverinfo='none',
        marker=dict(size=40, color='lightblue')  # Increase node size and set node color
    )

    for node in G.nodes(data=True):
        x, y = node_positions[node[0]]
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)
        node_trace['text'] += (G.nodes[node[0]]['label'],)

    fig = go.Figure(data=[temporal_edge_trace, interaction_edge_trace, node_trace],
                    layout=go.Layout(
                        title='Interactive Surgery Graph',
                        showlegend=False,
                        hovermode='closest',
                        xaxis=dict(showline=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showline=False, zeroline=False, showticklabels=False),
                        margin=dict(b=0, l=0, r=0, t=0),
                    ))
    fig.update_layout(annotations=annotations)
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    return fig


if __name__ == "__main__":
    file_path = 'data/llava_samples/surgery_sg.json'  # Replace with the path to your JSON file
    data = load_data(file_path)
    fig = create_interactive_graph(data)
    fig.show('chrome')
