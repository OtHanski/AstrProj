import matplotlib.pyplot as plt
import numpy as np

def load_tsv(file_path, min, max, binsize):
    bin_counts = np.loadtxt(file_path, delimiter='\t', skiprows=1)
    bins = np.arange(min, max + binsize, binsize)
    data = np.repeat(bins[:-1], bin_counts.astype(int))
    return data

def load_wi(file_path, min, max, binsize):
    bins = np.arange(min, max + binsize, binsize)
    data = np.loadtxt(file_path, delimiter='\t', skiprows=1)
    return data

def plot_histogram(data, label, color, ax, min, max, binsize):
    bins = np.arange(min, max + binsize, binsize)
    ax.hist(data, bins=bins, alpha=0.5, label=label, color=color)
    ax.legend(loc='upper right')

def plot_w_i(data, label, color, ax, min, max, binsize):
    bins = np.arange(min, max + binsize, binsize)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax.bar(bin_centers, data, width=binsize, alpha=0.5, label=label, color=color)
    ax.legend(loc='upper right')

def normalized_histogram(data, label, color, ax, min, max, binsize):
    bins = np.arange(min, max + binsize, binsize)
    counts, bin_edges = np.histogram(data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Avoid division by zero by adding a small constant
    sin_values = np.sin(np.deg2rad(bin_centers)) + 1e-10
    normalized_counts = counts / sin_values

    ax.bar(bin_centers, normalized_counts, width=binsize, alpha=0.5, label=label, color=color)
    ax.legend(loc='upper right')

def main():
    min = 0
    max = 180
    binsize = 0.25
    normalized = False

    data_DD = load_tsv('histogramDD.dat', min, max, binsize)
    data_DR = load_tsv('histogramDR.dat', min, max, binsize)
    data_RR = load_tsv('histogramRR.dat', min, max, binsize)
    data_wi = load_wi('w_i.dat', min, max, binsize)

    fig, axs = plt.subplots(3, 1, sharex=True, tight_layout=True)

    print("Histogram")
    plot_histogram(data_DD, 'DD', 'red', axs[0], min, max, binsize)
    plot_histogram(data_DR, 'DR', 'green', axs[1], min, max, binsize)
    plot_histogram(data_RR, 'RR', 'blue', axs[2], min, max, binsize)
    plt.show()  # Show the first figure

    fig, ax = plt.subplots(1, 1, tight_layout=True)  # Create a new figure for w_i
    print("w_i Histogram")
    print(data_wi)
    plot_w_i(data_wi, 'w_i', 'red', ax, min, max, binsize)

    plt.show()  # Show the second figure

    if normalized:
        fig, axs = plt.subplots(3, 1, sharex=True, tight_layout=True)
        print("Normalized Histogram")
        normalized_histogram(data_DD, 'DD', 'red', axs[0], min, max, binsize)
        normalized_histogram(data_DR, 'DR', 'green', axs[1], min, max, binsize)
        normalized_histogram(data_RR, 'RR', 'blue', axs[2], min, max, binsize)

        plt.show()

if __name__ == '__main__':
    main()