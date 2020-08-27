import matplotlib.pyplot as plt

from ODEs import ODEs
from clustering.ClusterSets.HierarchicalClusterSets import HierarchicalClusterSets
from clustering.ClusterSets.KMeansClusterSets import KMeansClusterSets
from clustering.Snapshots import Snapshots
from drawing.utils import display_name


def calculate_and_plot_silhouettes(
        axs,
        temporal_network,
        cluster_method,
        distance_metric,
        cluster_limit_type,
        cluster_limit_range,
        events,
        phases,
        variable_name,
        variable,
        time_ticks=None):

    (ax1, ax2, ax3) = axs

    try:
        snapshots = Snapshots.from_temporal_network(temporal_network, cluster_method, distance_metric)
        constructor = HierarchicalClusterSets if cluster_method != 'k_means' else KMeansClusterSets
        cluster_sets = constructor(snapshots, cluster_limit_type, cluster_limit_range)

        # Plot
        cluster_sets.plot_with_average_silhouettes((ax1, ax2, ax3))
        y_max = max(cluster_limit_range)
        y_min = min(cluster_limit_range)
        difference = y_max - y_min
        ax1.set_ylim(y_min - 0.2 * difference, y_max + 0.05 * difference)
        ODEs.plot_events(events, ax=ax1)
        ODEs.plot_phases(phases, ax=ax1, y_pos=0.05, ymax=0.1)

        # Format
        ax1.set_xlabel("Time")
        ax1.set_axisbelow(True)
        ax1.set_ylabel(display_name(cluster_sets.limit_type))
        if time_ticks:
            ax1.set_xticks(time_ticks)

        ax2.set_xlabel("Average silhouette")
        ax2.set_xlim((0, 1))
        ax2.yaxis.set_tick_params(labelleft=True)

        ax3.set_xlabel("Actual # clusters")
        max_size = max(cluster_sets.sizes)
        ax3.set_xlim(0, max_size + 0.5)
        ax3.set_xticks([2 * i for i in range((max_size + 1) // 2)])
        ax3.yaxis.set_tick_params(labelleft=True)

        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        fontdict = {'horizontalalignment': 'left'}
        ax1.set_title(f'Clusters and silhouette scores ({variable_name} = {variable})', fontdict=fontdict, pad=12)
        return True

    except Exception as e:
        print(f'Error when {variable_name} = {variable}: {e}')
        for ax in [ax1, ax2, ax3]:
            ax.clear()
        return False