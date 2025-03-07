import argparse
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.ticker import MaxNLocator

colorsMap = {
    (0, 0, 0, 4): 'red',
    (0, 1, 0, 4): 'blue',
    (0, 2, 0, 4): 'green',
    (1, 0, 0, 4): 'yellow',
    (1, 1, 0, 4): 'purple',
    (1, 2, 0, 4): 'orange',
    (0, 2, 1, 4): 'teal',
    (1, 2, 1, 4): 'maroon',
    (0, 0, 0, 7): 'brown',
    (0, 1, 0, 7): 'cyan',
    (0, 2, 0, 7): 'magenta',
    (1, 0, 0, 7): 'lime',
    (1, 1, 0, 7): 'olive',
    (1, 2, 0, 7): 'navy',
    (0, 2, 1, 7): 'gray',
    (1, 2, 1, 7): 'pink',
    (0, 0, 0, 3): 'indigo',
    (0, 1, 0, 3): 'violet',
    (0, 2, 0, 3): 'coral',
    (1, 0, 0, 3): 'aquamarine',
    (1, 1, 0, 3): 'lime',
    (1, 2, 0, 3): 'chocolate',
    (0, 2, 1, 3): 'crimson',
    (1, 2, 1, 3): 'darkblue',
    (0, 0, 0, 6): 'darkcyan',
    (0, 1, 0, 6): 'darkgreen',
    (0, 2, 0, 6): 'darkorange',
    (1, 0, 0, 6): 'deepskyblue',
    (1, 1, 0, 6): 'firebrick',
    (1, 2, 0, 6): 'fuchsia',
    (0, 2, 1, 6): 'khaki',
    (1, 2, 1, 6): 'lavender',
}

coll_name = ["Broadcast","Reduce","AllGather","ReduceScatter","AllReduce","SendRecv","Send","Recv","_","All2All","All2Allv"]

def drawLog(args):
    # Step 1: Open and read the log file
    with open(args.input, 'r') as f:
        lines = f.readlines()

    # Step 2: Parse the log file
    trace_data = {}
    statist_data = {}
    heatmap_data = {}
    for line in lines:
        _, workload_str, candidate_str, duration_str, repeat, _ = line.split(';')
        workload = tuple(map(int, workload_str.split()))
        candidate = tuple(map(int, candidate_str.split()))
        duration = float(duration_str)
        msg_bytes = workload[2]
        throughput = (msg_bytes/1024/1024/1024) / (duration/1000)

        if workload not in trace_data:
            trace_data[workload] = []
        trace_data[workload].append((candidate, throughput))

        # for statist
        if workload not in statist_data:
            statist_data[workload] = {}
        if candidate not in statist_data[workload]:
            statist_data[workload][candidate] = []
        statist_data[workload][candidate].append(throughput)

    for i, (workload, candidates_data) in enumerate(statist_data.items()):
        for j, (candidate, bw_list) in enumerate(candidates_data.items()):
            # for heatmap
            if workload not in heatmap_data:
                heatmap_data[workload] = {}
            implementation_wise = candidate[0:4]
            resource_wise = candidate[4:7]
            nc, nt, chunk = resource_wise
            if implementation_wise not in list(heatmap_data[workload]):
                heatmap_data[workload][implementation_wise] = {'nc': [], 'nt': [], 'chunk': [], 'bw': []}

            heatmap_data[workload][implementation_wise]['nc'].append(nc)
            heatmap_data[workload][implementation_wise]['nt'].append(nt)
            heatmap_data[workload][implementation_wise]['chunk'].append(chunk)
            def statis(bw_list, choose, percentile=None):
                array = np.array(bw_list)
                if choose == "median":
                    return np.median(array)
                elif choose == "mean":
                    return np.mean(array)
                elif choose == "percentile_mean" and percentile is not None:
                    lower_percentile = np.percentile(array, percentile[0])
                    upper_percentile = np.percentile(array, percentile[1])
                    filtered_array = array[(array >= lower_percentile) & (array <= upper_percentile)]
                    return np.mean(filtered_array)
                else:
                    raise ValueError("Invalid choice or missing percentile for percentile_mean")
            bw = statis(bw_list, "percentile_mean", (25, 75))
            heatmap_data[workload][implementation_wise]['bw'].append(bw)

    # Step 3: Create the plot
    if not args.subspace_heatmap:
        fig, axs = plt.subplots(len(trace_data), 1, figsize=(10, 2*len(trace_data)))
        for i, (workload, values) in enumerate(trace_data.items()):
            candidates, throughputs = zip(*values)
            if args.disable_highlight:
                # paint lines
                axs[i].plot(range(len(candidates)), throughputs, label=f'Workload {workload}')
            else:
                # paint colorful points for various subspaces
                prefixLength = 4
                colors = [colorsMap[tuple(candidate[:prefixLength])] for candidate in candidates]
                axs[i].scatter(range(len(candidates)), throughputs, color=colors, s=1)

            coll_type = workload[1]
            msg_bytes = workload[2]
            explain = ""
            if coll_name[coll_type] in ["AllGather", "ReduceScatter"]:
                explain = ", w/o *nrank"
            axs[i].set_title(f'Workload: {coll_name[coll_type]}, {msg_bytes} Bytes{explain}')
            axs[i].set_xlabel('Step')
            axs[i].set_ylabel('Algo. Bandwidth (GB/s)')
            # axs[i].legend()
        plt.tight_layout(pad=3.0)
    else:
        # for heatmap
        max_subspaces = max([len(value) for _, value in heatmap_data.items()])
        fig, axs = plt.subplots(len(heatmap_data), max_subspaces, figsize=(10, 2*len(heatmap_data)), subplot_kw={'projection': '3d'})
        if len(heatmap_data) == 1:
            axs = [axs]
        if max_subspaces == 1:
            axs = [[ax] for ax in axs]
        for row_idx, (workload, subspace_values) in enumerate(heatmap_data.items()):
            all_nc = []
            all_nt = []
            all_chunk = []
            max_subspace_count = 0
            for col_idx, (implementation_wise, values) in enumerate(subspace_values.items()):
                all_nc.extend(values['nc'])
                all_nt.extend(values['nt'])
                all_chunk.extend(values['chunk'])
                max_subspace_count = max(max_subspace_count, len(values))
            global_x_min, global_x_max = np.log2(min(all_nc)), np.log2(max(all_nc))
            global_y_min, global_y_max = np.log2(min(all_nt)), np.log2(max(all_nt))
            global_z_min, global_z_max = np.log2(min(all_chunk)), np.log2(max(all_chunk))
            for col_idx, (implementation_wise, values) in enumerate(subspace_values.items()):
                ax = axs[row_idx][col_idx]
                # data to numpy
                nc = np.array(values['nc'])
                log2_nc = np.log2(nc)
                nt = np.array(values['nt'])
                log2_nt = np.log2(nt)
                chunk = np.array(values['chunk'])
                log2_chunk = np.log2(chunk)
                bw = np.array(values['bw'])
                # scatter figure
                curr_subspace_count = len(values)
                dynamic_point_size = 50 * max_subspace_count/curr_subspace_count
                sc = ax.scatter(log2_nc, log2_nt, log2_chunk, c=bw, s=dynamic_point_size, cmap='viridis')
                last_scatter = sc  # update the last scatter

                ax.set_title(f'Subspace {implementation_wise}')
                ax.set_xlabel('nc')
                ax.set_ylabel('nt')
                ax.set_zlabel('chunk')

                # set the tikcs to show the original value
                ax.set_xticks(log2_nc)
                ax.set_xticklabels(nc)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.set_yticks(log2_nt)
                ax.set_yticklabels(nt)
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                ax.set_zticks(log2_chunk)
                ax.set_zticklabels(chunk)
                ax.zaxis.set_major_locator(MaxNLocator(integer=True))

                # share the x, y, z range
                ax.set_xlim(global_x_min, global_x_max)
                ax.set_ylim(global_y_min, global_y_max)
                ax.set_zlim(global_z_min, global_z_max)

            # add a colorbar for each row
            if last_scatter is not None:
                cbar = fig.colorbar(last_scatter, ax=axs[row_idx], orientation='vertical', shrink=0.5, aspect=5)
                cbar.set_label('Bandwidth')

            coll_type = workload[1]
            msg_bytes = workload[2]
            explain = ""
            if coll_name[coll_type] in ["AllGather", "ReduceScatter"]:
                explain = ", w/o *nrank"
            # add a suptitle for each row
            fig.suptitle(f'Workload: {coll_name[coll_type]}, {msg_bytes} Bytes{explain}', y=1.02)

            # hidden the empty subspaces
            for col_idx in range(len(subspace_values), max_subspaces):
                fig.delaxes(axs[row_idx][col_idx])

        plt.tight_layout()
    plt.savefig(args.output+'tccl_log.pdf')  # Save the figure before showing it
    plt.show()  # Show the figure


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="input")
    parser.add_argument("--output", type=str, default="./", help="output")
    parser.add_argument("--disable_highlight", action='store_true', default=False, help="disable highlight for various subspace")
    parser.add_argument("--subspace_heatmap", action='store_true', default=False, help="heatmap for various subspace")

    args = parser.parse_args()

    drawLog(args)
