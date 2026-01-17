import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_bpp_vs_mssim(df, save_path='bpp_vs_mssim.png'):
    """
    Plots MS-SSIM vs. BPP for different codecs.
    """
    # Filter data for each algorithm
    mjpeg_df = df[df['Algorithm'].str.startswith('MJPEG')].sort_values(by='Avg BPP')
    h264_df = df[df['Algorithm'].str.startswith('H264')].sort_values(by='Avg BPP')
    mcucoder_df = df[df['Algorithm'].str.startswith('MCUCoder')].sort_values(by='Avg BPP')

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot data for each algorithm
    plt.plot(mjpeg_df['Avg BPP'], mjpeg_df['Avg MS-SSIM'], marker='o', linestyle='-', label='MJPEG')
    plt.plot(h264_df['Avg BPP'], h264_df['Avg MS-SSIM'], marker='s', linestyle='-', label='H264')
    plt.plot(mcucoder_df['Avg BPP'], mcucoder_df['Avg MS-SSIM'], marker='^', linestyle='-', label='MCUCoder')

    # Add titles and labels
    plt.title('MS-SSIM vs. BPP for different codecs')
    plt.xlabel('Average BPP (bits per pixel)')
    plt.ylabel('Average MS-SSIM')
    plt.grid(True)
    plt.legend()

    # Save the plot
    plt.savefig(save_path)

    print(f"Plot saved as {save_path}")

def plot_bpp_vs_psnr(df, save_path='bpp_vs_psnr.png'):
    """
    Plots PSNR vs. BPP for different codecs.
    """
    # Filter data for each algorithm
    mjpeg_df = df[df['Algorithm'].str.startswith('MJPEG')].sort_values(by='Avg BPP')
    h264_df = df[df['Algorithm'].str.startswith('H264')].sort_values(by='Avg BPP')
    mcucoder_df = df[df['Algorithm'].str.startswith('MCUCoder')].sort_values(by='Avg BPP')

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot data for each algorithm
    plt.plot(mjpeg_df['Avg BPP'], mjpeg_df['Avg PSNR'], marker='o', linestyle='-', label='MJPEG')
    plt.plot(h264_df['Avg BPP'], h264_df['Avg PSNR'], marker='s', linestyle='-', label='H264')
    plt.plot(mcucoder_df['Avg BPP'], mcucoder_df['Avg PSNR'], marker='^', linestyle='-', label='MCUCoder')

    # Add titles and labels
    plt.title('PSNR vs. BPP for different codecs')
    plt.xlabel('Average BPP (bits per pixel)')
    plt.ylabel('Average PSNR (dB)')
    plt.grid(True)
    plt.legend()

    # Save the plot
    plt.savefig(save_path)

    print(f"Plot saved as {save_path}")

def plot_memory_footprint(df, save_path='memory_footprint.png'):
    """
    Plots a horizontal bar chart of the peak encoder RAM usage.
    """
    # Find the maximum RAM usage for each algorithm group
    mjpeg_ram = df[df['Algorithm'].str.startswith('MJPEG')]['Peak Enc RAM (KB)'].max()
    h264_ram = df[df['Algorithm'].str.startswith('H264')]['Peak Enc RAM (KB)'].max()
    mcucoder_ram = df[df['Algorithm'].str.startswith('MCUCoder')]['Peak Enc RAM (KB)'].max()

    # Create a new DataFrame for plotting
    ram_data = {
        'Algorithm': ['MJPEG', 'H264', 'MCUCoder'],
        'Peak Enc RAM (KB)': [mjpeg_ram, h264_ram, mcucoder_ram]
    }
    ram_df = pd.DataFrame(ram_data).sort_values(by='Peak Enc RAM (KB)', ascending=True)

    # Create the plot
    plt.figure(figsize=(10, 6))
    bars = plt.barh(ram_df['Algorithm'], ram_df['Peak Enc RAM (KB)'])
    
    # Add a vertical line for a typical microcontroller's RAM limit
    esp32_ram_limit = 520  # ESP32 has 520 KB of SRAM
    plt.axvline(x=esp32_ram_limit, color='r', linestyle='--', label=f'ESP32 Limit ({esp32_ram_limit}KB)')

    # Add data labels to the bars
    for bar in bars:
        plt.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
                 f'{bar.get_width():.0f} KB',
                 va='center', ha='left')

    # Add titles and labels
    plt.title('Hardware Cost: Memory Footprint')
    plt.xlabel('Peak Encoder RAM (KB) - Lower is Better')
    plt.ylabel('Algorithm')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.legend()
    
    # Adjust plot limits to make space for labels
    plt.xlim(right=ram_df['Peak Enc RAM (KB)'].max() * 1.15)
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path)
    print(f"Plot saved as {save_path}")

def plot_encoding_time_boxplot(df, save_path='encoding_time_boxplot.png'):
    """
    Plots a box plot of the encoding time for each algorithm.
    """
    # Add a 'Codec' column for grouping
    df['Codec'] = df['Algorithm'].apply(lambda x: x.split('_')[0])

    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Create the boxplot
    df.boxplot(column='Avg Enc Time', by='Codec', grid=True)

    # Add titles and labels
    plt.title('Encoding Time Distribution by Codec')
    plt.suptitle('') # Suppress the default title
    plt.xlabel('Codec')
    plt.ylabel('Average Encoding Time (s)')
    
    # Save the plot
    plt.savefig(save_path)
    print(f"Plot saved as {save_path}")

def plot_radar_chart(df, save_path='radar_chart.png'):
    """
    Creates a radar chart to compare codecs across multiple metrics.
    """
    # Define metrics and whether higher is better
    metrics = {
        'Avg BPP': False,
        'Avg MS-SSIM': True,
        'Avg PSNR': True,
        'Avg Enc Time': False,
        'Peak Enc RAM (KB)': False
    }
    
    df['Codec'] = df['Algorithm'].apply(lambda x: x.split('_')[0])
    
    # Find the row with the lowest BPP for each codec
    lowest_bpp_df = df.loc[df.groupby('Codec')['Avg BPP'].idxmin()]
    
    # For 'Peak Enc RAM (KB)', find the smallest non-zero value for each codec
    # for codec in lowest_bpp_df['Codec'].unique():
    #     min_ram = df[(df['Codec'] == codec) & (df['Peak Enc RAM (KB)'] > 0)]['Peak Enc RAM (KB)'].min()
    #     if pd.notna(min_ram):
    #         lowest_bpp_df.loc[lowest_bpp_df['Codec'] == codec, 'Peak Enc RAM (KB)'] = min_ram

    # Set 'Codec' as the index for easy lookup and select metrics
    grouped = lowest_bpp_df.set_index('Codec')[list(metrics.keys())]

    # Normalize the data
    normalized_df = pd.DataFrame(index=grouped.index)
    for metric, higher_is_better in metrics.items():
        min_val = grouped[metric].min()
        max_val = grouped[metric].max()
        if max_val == min_val:
            normalized_df[metric] = 0.5 # Assign a neutral value if all values are the same
        else:
            if higher_is_better:
                normalized_df[metric] = (grouped[metric] - min_val) / (max_val - min_val)
            else: # Lower is better
                normalized_df[metric] = 1 - ((grouped[metric] - min_val) / (max_val - min_val))

    # Create the radar chart
    labels = normalized_df.columns
    num_vars = len(labels)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] # Complete the loop

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for i, row in normalized_df.iterrows():
        data = row.tolist()
        data += data[:1] # Complete the loop
        ax.plot(angles, data, label=i)
        ax.fill(angles, data, alpha=0.25)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    plt.title('Codec Performance Radar Chart', size=20, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Plot saved as {save_path}")

if __name__ == "__main__":
    # Load the data from the CSV file
    input = "/mnt/c/Users/User/Desktop/bku/251/benchmark_compression/dancing/benchmark_summary.csv"
    output= "/mnt/c/Users/User/Desktop/bku/251/benchmark_compression/dancing/"
    df = pd.read_csv(input)

    # Generate the plots
    plot_bpp_vs_mssim(df, save_path=output + 'bpp_vs_mssim.png')
    plot_bpp_vs_psnr(df, save_path=output + 'bpp_vs_psnr.png')
    plot_memory_footprint(df, save_path=output + 'memory_footprint.png')
    plot_encoding_time_boxplot(df, save_path=output + 'encoding_time_boxplot.png')
    plot_radar_chart(df, save_path=output + 'radar_chart.png')
