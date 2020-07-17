import pandas as pd
import teneto

# ToDo: Wrapper class around teneto network? Expose useful properties/methods?


def temporal_network_from_file(filepath, separator):
    # edges are of type pandas.DataFrame
    edges = pd.read_csv(filepath, sep=separator, engine='python')

    # ToDo: is this necessary??
    # Columns must be named i, j and t, with optional weight column
    number_of_columns = edges.shape[1]
    if number_of_columns == 3:
        columns = ['i', 'j', 't']
    elif number_of_columns == 4:
        columns = ['i', 'j', 't', 'weight']
    else:
        raise ValueError('List of edges requires either 3 or 4 columns')
    edges.columns = columns

    # Replace strings with numeric values
    nodes = sorted(set(edges[['i', 'j']].values.flatten()))
    node_ids = {key: i for i, key in enumerate(nodes)}
    edges = edges.replace(node_ids)

    times = sorted(set(edges['t']))
    start_time = times[0]
    # For compatibility with teneto, shift all times so that we start at time 0.
    # ToDo: find a neat way to undo this effect anytime we present the data?
    edges['t'] -= start_time

    return teneto.TemporalNetwork(from_df=edges, nodelabels=nodes)
