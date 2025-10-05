import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from matplotlib.collections import LineCollection
from scipy.signal import find_peaks
from matplotlib.patches import Ellipse


# 1. 
def load_worm_data(file_path, fps=2.5):
    """"""
    df = pd.read_csv(file_path, sep='\t')

   
    required_columns = ['frame', 'x1', 'y1']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"————————————————————————-: {required_columns}")

    
    df['time'] = df['frame'] / fps  



    return df



# 2. 
def calculate_movement_parameters(df, worm_id=None, smoothing_window=5, fps=2.5):
    """"""
    if worm_id is not None:
        df = df[df['worm_id'] == worm_id].copy()


    df = df.sort_values('frame').reset_index(drop=True)

    df['dx'] = df['x1'].diff().fillna(0)
    df['dy'] = df['y1'].diff().fillna(0)
    df['displacement'] = np.sqrt(df['dx'] ** 2 + df['dy'] ** 2)

 
    time_diff = df['time'].diff().replace(0, np.nan)
    df['speed'] = (df['displacement'] / time_diff).fillna(0)


    df['direction'] = np.arctan2(df['dy'], df['dx'].replace(0, 1e-10)) 


    df['speed_smoothed'] = df['speed'].copy()
    df['direction_smoothed'] = df['direction'].copy()

    if len(df) > smoothing_window:
        try:
            valid_window = min(smoothing_window, len(df) - 1)
            valid_window = valid_window if valid_window % 2 == 1 else valid_window - 1
            if valid_window > 2:
  
                df['speed_smoothed'] = savgol_filter(
                    df['speed'].clip(0, None).fillna(0),  
                    window_length=valid_window,
                    polyorder=2,
                    mode='interp'  
                )
                df['direction_smoothed'] = savgol_filter(
                    df['direction'].fillna(0),
                    window_length=valid_window,
                    polyorder=2,
                    mode='interp'
                )
        except Exception as e:

            df['speed_smoothed'] = df['speed'].rolling(5, min_periods=1).mean()
            df['direction_smoothed'] = df['direction'].rolling(5, min_periods=1).mean()


    df['movement_state'] = 'stationary'
    if len(df) > 1:

        speed_threshold = max(df['speed_smoothed'].quantile(0.1), 0.5) 

        for i in range(1, len(df)):
            if df['speed_smoothed'].iloc[i] > speed_threshold:
                angle_diff = abs(df['direction_smoothed'].iloc[i] - df['direction_smoothed'].iloc[i - 1])
                angle_diff = min(angle_diff, 2 * np.pi - angle_diff)

                if angle_diff > np.pi / 2 and df['speed_smoothed'].iloc[i] > df['speed_smoothed'].iloc[i - 1]:
                    df.at[i, 'movement_state'] = 'backward'
                else:
                    df.at[i, 'movement_state'] = 'forward'

    return df


def plot_movement_analysis(worm_df):

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))


    if not worm_df.empty:
        segments = np.array([
            [[x1, y1], [x2, y2]]
            for (x1, y1), (x2, y2) in zip(
                worm_df[['x1', 'y1']].values[:-1],
                worm_df[['x1', 'y1']].values[1:]
            )
        ])
        lc = LineCollection(
            segments,
            cmap='viridis',
            norm=plt.Normalize(worm_df['speed_smoothed'].min(), worm_df['speed_smoothed'].max())
        )
        lc.set_array(worm_df['speed_smoothed'])
        ax1.add_collection(lc)
        ax1.set_xlim(worm_df['x1'].min() - 10, worm_df['x1'].max() + 10)
        ax1.set_ylim(worm_df['y1'].min() - 10, worm_df['y1'].max() + 10)
        fig.colorbar(lc, ax=ax1, label='Speed (pixels/sec)')


    ax2.plot(worm_df['time'], worm_df['speed'], 'gray', alpha=0.3, label='Raw')
    ax2.plot(worm_df['time'], worm_df['speed_smoothed'], 'b-', label='Smoothed')
    ax2.set_ylim(bottom=0, top=max(1, worm_df['speed_smoothed'].max() * 1.1))
    ax2.legend()
    ax2.grid(True)

    colors = {'forward': 'green', 'backward': 'red', 'stationary': 'gray'}
    for state, color in colors.items():
        mask = worm_df['movement_state'] == state
        ax3.scatter(
            worm_df['time'][mask],
            np.ones(sum(mask)) if state == 'stationary' else worm_df['speed_smoothed'][mask],
            c=color,
            label=state,
            s=20,
            alpha=0.7
        )
    ax3.set_yticks([0, 1, 2])
    ax3.set_yticklabels(['', 'Moving', ''])
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    return fig


def detect_omega_turns(worm_df, fps=2.5):

    dx = np.gradient(worm_df['x1'].values)
    dy = np.gradient(worm_df['y1'].values)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = (dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** 1.5


    omega_threshold = np.percentile(curvature, 90)  
    omega_frames, _ = find_peaks(np.abs(curvature), height=omega_threshold)


    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(worm_df['time'], curvature, label='Curvature')
    ax.scatter(worm_df['time'].iloc[omega_frames], curvature[omega_frames],
               c='red', label='Omega Turns')
    ax.axhline(omega_threshold, linestyle='--', color='gray')
    ax.set_title(f'Omega Turn Detection (Found {len(omega_frames)} events)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Curvature')
    ax.legend()
    plt.tight_layout()

    return {
        'count': len(omega_frames),
        'frames': omega_frames,
        'curvature': curvature,
        'figure': fig
    }



def analyze_undulation(worm_df, fps=2.5):


    widths = worm_df['w'].values
    peaks, _ = find_peaks(widths, prominence=np.std(widths) / 2)

    if len(peaks) > 1:
        duration = len(widths) / fps
        freq = len(peaks) / duration  # Hz
    else:
        freq = 0

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(worm_df['time'], widths, label='Body Width')
    ax.scatter(worm_df['time'].iloc[peaks], widths[peaks], c='red', label='Peaks')
    ax.set_title(f'Undulation Frequency: {freq:.2f} Hz')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Width (pixels)')
    ax.legend()
    plt.tight_layout()

    return {
        'frequency': freq,
        'peaks': peaks,
        'figure': fig
    }

def detect_rolls(worm_df):

    ratios = (worm_df['w'] / worm_df['h']).values
    roll_threshold = 1.5  
    roll_frames = np.where(np.diff(ratios) > roll_threshold)[0]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(worm_df['time'], ratios, label='Width/Height Ratio')
    ax.scatter(worm_df['time'].iloc[roll_frames], ratios[roll_frames],
               c='red', label='Roll Events')
    ax.axhline(np.mean(ratios), linestyle='--', color='gray', label='Mean Ratio')
    ax.set_title(f'Roll Detection (Found {len(roll_frames)} events)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Width/Height Ratio')
    ax.legend()
    plt.tight_layout()

    return {
        'count': len(roll_frames),
        'frames': roll_frames,
        'ratios': ratios,
        'figure': fig
    }




def plot_behavior_summary(worm_df, behaviors):

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))


    segments = np.array([[[x1, y1], [x2, y2]] for (x1, y1), (x2, y2) in
                         zip(worm_df[['x1', 'y1']].values[:-1], worm_df[['x1', 'y1']].values[1:])])
    lc = LineCollection(segments, cmap='viridis', norm=plt.Normalize(0, worm_df['speed_smoothed'].max()))
    lc.set_array(worm_df['speed_smoothed'].values[:-1])
    ax1.add_collection(lc)


    omega_frames = behaviors['omega']['frames']
    ax1.scatter(worm_df['x1'].iloc[omega_frames], worm_df['y1'].iloc[omega_frames],
                c='red', s=100, label='Omega Turns')


    roll_frames = behaviors['roll']['frames']
    ax1.scatter(worm_df['x1'].iloc[roll_frames], worm_df['y1'].iloc[roll_frames],
                c='blue', marker='x', s=100, label='Rolls')

    ax1.set_xlim(worm_df['x1'].min() - 10, worm_df['x1'].max() + 10)
    ax1.set_ylim(worm_df['y1'].min() - 10, worm_df['y1'].max() + 10)
    ax1.set_title('Behavior Events on Trajectory')
    ax1.legend()

    ax2.plot(worm_df['time'], worm_df['speed_smoothed'], label='Speed')
    ax2.scatter(worm_df['time'].iloc[omega_frames], worm_df['speed_smoothed'].iloc[omega_frames],
                c='red', label='Omega')
    ax2.scatter(worm_df['time'].iloc[roll_frames], worm_df['speed_smoothed'].iloc[roll_frames],
                c='blue', marker='x', label='Roll')
    ax2.set_ylabel('Speed (pixels/s)')
    ax2.legend()


    ax3.plot(worm_df['time'], worm_df['w'], label='Body Width')
    ax3.scatter(worm_df['time'].iloc[behaviors['undulation']['peaks']],
                worm_df['w'].iloc[behaviors['undulation']['peaks']],
                c='green', label='Undulation Peaks')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Width (pixels)')
    ax3.legend()

    plt.tight_layout()
    return fig


def plot_behavior_heatmap(worm_df, behaviors, bin_size=20):

    x_bins = np.arange(worm_df['x1'].min(), worm_df['x1'].max() + bin_size, bin_size)
    y_bins = np.arange(worm_df['y1'].min(), worm_df['y1'].max() + bin_size, bin_size)

    omega_density = np.zeros((len(y_bins) - 1, len(x_bins) - 1))
    roll_density = np.zeros_like(omega_density)
    undulation_density = np.zeros_like(omega_density)


    for i in range(len(x_bins) - 1):
        for j in range(len(y_bins) - 1):
            mask = (worm_df['x1'] >= x_bins[i]) & (worm_df['x1'] < x_bins[i + 1]) & \
                   (worm_df['y1'] >= y_bins[j]) & (worm_df['y1'] < y_bins[j + 1])


            omega_mask = np.isin(worm_df.index, behaviors['omega']['frames'])
            omega_density[j, i] = np.sum(mask & omega_mask)

            roll_mask = np.isin(worm_df.index, behaviors['roll']['frames'])
            roll_density[j, i] = np.sum(mask & roll_mask)


            undulation_density[j, i] = np.mean(worm_df.loc[mask, 'w'].diff().abs()) if np.any(mask) else 0

 
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    im1 = ax1.imshow(omega_density, cmap='Reds',
                     extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]],
                     origin='lower', aspect='auto')
    ax1.set_title('Omega Turns Density')
    fig.colorbar(im1, ax=ax1, label='Event Count')

 
    im2 = ax2.imshow(roll_density, cmap='Blues',
                     extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]],
                     origin='lower', aspect='auto')
    ax2.set_title('Roll Events Density')
    fig.colorbar(im2, ax=ax2, label='Event Count')

  
    im3 = ax3.imshow(undulation_density, cmap='Greens',
                     extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]],
                     origin='lower', aspect='auto')
    ax3.set_title('Undulation Intensity')
    fig.colorbar(im3, ax=ax3, label='Width Variation')


    for ax in [ax1, ax2, ax3]:
        ax.plot(worm_df['x1'], worm_df['y1'], 'k-', lw=0.5, alpha=0.3)
        ax.set_xlabel('X position')
    ax1.set_ylabel('Y position')

    plt.tight_layout()
    return fig



if __name__ == "__main__":

    FPS = 2.5  
    TARGET_WORM_ID = 2  


    data = load_worm_data('worm_tracking_results.txt', fps=FPS)

    worm_data = calculate_movement_parameters(data, worm_id=TARGET_WORM_ID, fps=FPS)


    fig = plot_movement_analysis(worm_data)
    output_path = f'worm_{TARGET_WORM_ID}_movement_analysis_{FPS}fps.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n {output_path}")
    plt.show()

    behaviors = {
        'omega': detect_omega_turns(worm_data, fps=FPS),
        'undulation': analyze_undulation(worm_data, fps=FPS),
        'roll': detect_rolls(worm_data)
    }


    for name, result in behaviors.items():
        result['figure'].savefig(f'worm_{TARGET_WORM_ID}_{name}.png', dpi=300)
        plt.close(result['figure'])

    summary_fig = plot_behavior_summary(worm_data, behaviors)
    summary_fig.savefig(f'worm_{TARGET_WORM_ID}_summary.png', dpi=300)

    print("\n===== Behavior Analysis Report =====")
    print(f"Omega Turns: {behaviors['omega']['count']} events")
    print(f"Undulation Frequency: {behaviors['undulation']['frequency']:.2f} Hz")
    print(f"Roll Events: {behaviors['roll']['count']} times")

    heatmap_fig = plot_behavior_heatmap(worm_data, behaviors, bin_size=15)
    heatmap_fig.savefig(f'worm_{TARGET_WORM_ID}_heatmap.png', dpi=300)
    plt.close(heatmap_fig)

    print("\n===== Enhanced Behavior Report =====")
    print(f"Omega Turns Hotspots: {np.max(behaviors['omega']['density'])} events/bin")
    print(f"Roll Events Hotspots: {np.max(behaviors['roll']['density'])} events/bin")
    print(f"Undulation Intensity Peak: {np.max(behaviors['undulation']['intensity']):.2f} px variation")