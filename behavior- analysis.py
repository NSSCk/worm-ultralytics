import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from matplotlib.collections import LineCollection


# 1. 
def load_worm_data(file_path, fps=2.5):
    """"""
    df = pd.read_csv(file_path, sep='\t')

    # 
    required_columns = ['frame', 'x1', 'y1']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f" {required_columns}")

    # 
    df['time'] = df['frame'] / fps  # 



    return df


# 2.


# def calculate_movement_parameters(df, worm_id=None, smoothing_window=5, fps=2.5):
#     """"""
#     if worm_id is not None:
#         df = df[df['worm_id'] == worm_id].copy()
#         print(f" ID: {worm_id}...")
#
#     # 
#     df = df.sort_values('frame').reset_index(drop=True)
#
#     # 
#     df['dx'] = df['x1'].diff()
#     df['dy'] = df['y1'].diff()
#     df['displacement'] = np.sqrt(df['dx'] ** 2 + df['dy'] ** 2)
#     time_diff = df['time'].diff().replace(0, np.nan)
#     df['speed'] = df['displacement'] / time_diff
#     df['speed'] = df['speed'].fillna(0)
#
#     # 
#     df['direction'] = np.arctan2(df['dy'], df['dx'])
#
#     # 
#     df['speed_smoothed'] = df['speed'].copy()
#     df['direction_smoothed'] = df['direction'].copy()  # 
#
#     if len(df) > smoothing_window:
#         try:
#             valid_window = min(smoothing_window, len(df) - 1)
#             valid_window = valid_window if valid_window % 2 == 1 else valid_window - 1
#             if valid_window > 2:
#                 # 
#                 df['speed_smoothed'] = savgol_filter(
#                     df['speed'].fillna(0),
#                     window_length=valid_window,
#                     polyorder=2,
#                     mode='nearest'
#                 )
#                 df['direction_smoothed'] = savgol_filter(
#                     df['direction'].fillna(0),
#                     window_length=valid_window,
#                     polyorder=2,
#                     mode='nearest'
#                 )
#              
#         except Exception as e:
#            
#             df['speed_smoothed'] = df['speed'].rolling(3, min_periods=1).mean()
#             df['direction_smoothed'] = df['direction'].rolling(3, min_periods=1).mean()
#
#     # 
#     df['movement_state'] = 'stationary'
#     if len(df) > 1:
#         speed_threshold = max(df['speed_smoothed'].quantile(0.25), 0.1)
#
#         for i in range(1, len(df)):
#             if df['speed_smoothed'].iloc[i] > speed_threshold:
#                 angle_diff = abs(df['direction_smoothed'].iloc[i] - df['direction_smoothed'].iloc[i - 1])
#                 angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
#                 df.at[i, 'movement_state'] = 'backward' if angle_diff > np.pi / 2 else 'forward'
#

#
#     return df

# 3. 
# def plot_movement_analysis(worm_df, worm_id=None):
#   
#     fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
#
#     # 
#     points = np.array([worm_df['x1'], worm_df['y1']]).T.reshape(-1, 1, 2)
#     segments = np.concatenate([points[:-1], points[1:]], axis=1)
#     norm = plt.Normalize(worm_df['speed_smoothed'].min(), worm_df['speed_smoothed'].max())
#     lc = LineCollection(segments, cmap='viridis', norm=norm)
#     lc.set_array(worm_df['speed_smoothed'])
#     ax1.add_collection(lc)
#     ax1.set_xlim(worm_df['x1'].min() - 10, worm_df['x1'].max() + 10)
#     ax1.set_ylim(worm_df['y1'].min() - 10, worm_df['y1'].max() + 10)
#     ax1.set_title(f'Worm {worm_id} Trajectory (Color = Speed)' if worm_id else 'Movement Trajectory')
#     fig.colorbar(lc, ax=ax1, label='Speed (pixels/sec)')
#
#     # 
#     ax2.plot(worm_df['time'], worm_df['speed'], 'gray', alpha=0.3, label='')
#     ax2.plot(worm_df['time'], worm_df['speed_smoothed'], 'b', label='')
#     ax2.set_ylabel('Speed (pixels/sec)')
#     ax2.legend()
#     ax2.grid(True)
#
#     # 
#     colors = {'forward': 'green', 'backward': 'red', 'stationary': 'gray'}
#     for state, color in colors.items():
#         mask = worm_df['movement_state'] == state
#         ax3.scatter(worm_df['time'][mask], worm_df['speed_smoothed'][mask],
#                     c=color, label=state, s=15)
#     ax3.set_xlabel('Time (seconds)')
#     ax3.set_ylabel('Movement State')
#     ax3.legend()
#     ax3.grid(True)
#
#     plt.tight_layout()
#     return fig
def calculate_movement_parameters(df, worm_id=None, smoothing_window=5, fps=2.5):
    """"""
    if worm_id is not None:
        df = df[df['worm_id'] == worm_id].copy()

    # 
    df = df.sort_values('frame').reset_index(drop=True)

    # 
    df['dx'] = df['x1'].diff().fillna(0)
    df['dy'] = df['y1'].diff().fillna(0)
    df['displacement'] = np.sqrt(df['dx'] ** 2 + df['dy'] ** 2)

    # 
    time_diff = df['time'].diff().replace(0, np.nan)
    df['speed'] = (df['displacement'] / time_diff).fillna(0)

    # 
    df['direction'] = np.arctan2(df['dy'], df['dx'].replace(0, 1e-10))  # 

    # 
    df['speed_smoothed'] = df['speed'].copy()
    df['direction_smoothed'] = df['direction'].copy()

    if len(df) > smoothing_window:
        try:
            valid_window = min(smoothing_window, len(df) - 1)
            valid_window = valid_window if valid_window % 2 == 1 else valid_window - 1
            if valid_window > 2:
                #
                df['speed_smoothed'] = savgol_filter(
                    df['speed'].clip(0, None).fillna(0),  # 
                    window_length=valid_window,
                    polyorder=2,
                    mode='interp'  # 
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

    # 
    df['movement_state'] = 'stationary'
    if len(df) > 1:
        # 
        speed_threshold = max(df['speed_smoothed'].quantile(0.1), 0.5)  # 

        for i in range(1, len(df)):
            if df['speed_smoothed'].iloc[i] > speed_threshold:
                angle_diff = abs(df['direction_smoothed'].iloc[i] - df['direction_smoothed'].iloc[i - 1])
                angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
                # 
                if angle_diff > np.pi / 2 and df['speed_smoothed'].iloc[i] > df['speed_smoothed'].iloc[i - 1]:
                    df.at[i, 'movement_state'] = 'backward'
                else:
                    df.at[i, 'movement_state'] = 'forward'

    return df


def plot_movement_analysis(worm_df):
    """"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

    # 
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

    # 
    ax2.plot(worm_df['time'], worm_df['speed'], 'gray', alpha=0.3, label='Raw')
    ax2.plot(worm_df['time'], worm_df['speed_smoothed'], 'b-', label='Smoothed')
    ax2.set_ylim(bottom=0, top=max(1, worm_df['speed_smoothed'].max() * 1.1))
    ax2.legend()
    ax2.grid(True)

    # 
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

# 4. 
if __name__ == "__main__":
    # 
    FPS = 2.5  # 
    TARGET_WORM_ID = 8  # 

    # 
    data = load_worm_data('worm_tracking_results.txt', fps=FPS)

    # 
    worm_data = calculate_movement_parameters(data, worm_id=TARGET_WORM_ID, fps=FPS)

    # 
    fig = plot_movement_analysis(worm_data)
    output_path = f'worm_{TARGET_WORM_ID}_movement_analysis_{FPS}fps.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
 
    plt.show()
    #

