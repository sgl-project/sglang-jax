import argparse
from io import StringIO

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SGLang JAX CI Performance (Perf) Trend Visualization Tool"
    )

    # --- Basic Configuration ---
    parser.add_argument(
        "--owner", type=str, default="pathfinder-pf", help="GitHub repository owner"
    )
    parser.add_argument(
        "--repo", type=str, default="sglang-jax-ci-data", help="GitHub repository name"
    )
    parser.add_argument(
        "--subdir", type=str, default="perf", help="Subdirectory containing the data"
    )

    # Modify default filename to perf-related csv (adjust based on actual situation)
    parser.add_argument(
        "--file",
        type=str,
        default="performance_results_BAILING_MOE_tp_2_ep_2.csv",
        help="Target CSV filename",
    )

    # --- Critical: Performance Hyperparameter Filtering ---
    parser.add_argument("--concurrency", type=int, default=8, help="Filter concurrency")
    parser.add_argument("--input", type=int, default=1024, help="Filter input length")
    parser.add_argument("--output", type=int, default=1024, help="Filter output length")

    # --- Limit Data Points ---
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=0,
        help="Maximum number of recent data points to fetch (0 means no limit)",
    )

    # --- Network Configuration ---
    parser.add_argument("--proxy", type=str, default=None, help="HTTP proxy address")
    parser.add_argument("--token", type=str, default=None, help="GitHub Token")

    # --- Output Configuration ---
    parser.add_argument(
        "--output_file", type=str, default="perf_trend.png", help="Output image filename"
    )

    return parser.parse_args()


def get_session(proxy=None, token=None):
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    if proxy:
        session.proxies = {"http": proxy, "https": proxy}
    headers = {"User-Agent": "Mozilla/5.0"}
    if token:
        headers["Authorization"] = f"token {token}"
    session.headers.update(headers)
    return session


def get_date_folders(session, args):
    api_url = f"https://api.github.com/repos/{args.owner}/{args.repo}/contents/{args.subdir}"
    try:
        response = session.get(api_url, timeout=30)
        response.raise_for_status()
        # Get all directories and sort them
        all_dates = sorted([item["name"] for item in response.json() if item["type"] == "dir"])

        if args.limit > 0 and len(all_dates) > args.limit:
            print(
                f"Detected {len(all_dates)} historical data points, taking only the recent {args.limit}."
            )
            all_dates = all_dates[-args.limit :]
        else:
            print(f"Processing all {len(all_dates)} historical data points.")

        return all_dates
    except Exception as e:
        print(f"Failed to fetch directories: {e}")
        return []


def fetch_data(session, date_folders, args):
    all_rows = []
    print("Downloading and merging data...")
    for date_str in date_folders:
        raw_url = f"https://raw.githubusercontent.com/{args.owner}/{args.repo}/main/{args.subdir}/{date_str}/{args.file}"
        try:
            res = session.get(raw_url, timeout=30)
            if res.status_code == 200:
                df = pd.read_csv(StringIO(res.text))
                df["Date_Str"] = date_str
                all_rows.append(df)
        except Exception:
            pass
    return pd.concat(all_rows, ignore_index=True) if all_rows else None


def plot_perf_results(df, args):
    """
    Visualize performance data: filter for specific configurations and plot a 2x2 grid of metrics.
    """
    if df is None or df.empty:
        print("No data available to plot")
        return

    # 1. Data Cleaning: Convert relevant columns to numeric
    numeric_cols = [
        "concurrency",
        "input",
        "output",
        "ttft_ms",
        "itl_ms",
        "in_tps",
        "out_tps",
        "tpu_size",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2. Core Logic: Filter data based on hyperparameters
    # Performance results usually contain multiple configurations; we must filter for a unique set to plot a trend.
    filter_condition = (
        (df["concurrency"] == args.concurrency)
        & (df["input"] == args.input)
        & (df["output"] == args.output)
    )
    filtered_df = df[filter_condition].copy()

    if filtered_df.empty:
        print(
            f"Warning: No data found for current filters (Concurrency={args.concurrency}, Input={args.input}, Output={args.output})"
        )
        print("Please check if the CSV file configurations match the arguments.")
        return

    # Sort by date
    filtered_df = filtered_df.sort_values("Date_Str")

    # Create legend labels: Model Name + TPU Size (if available)
    if "tpu_size" in filtered_df.columns:
        filtered_df["Legend_Entry"] = (
            filtered_df["model_name"] + " (TPU-" + filtered_df["tpu_size"].astype(str) + ")"
        )
    else:
        filtered_df["Legend_Entry"] = filtered_df["model_name"]

    # 3. Prepare X-axis mapping
    unique_dates = filtered_df["Date_Str"].unique()
    date_map = {date: i for i, date in enumerate(unique_dates)}
    filtered_df["x_idx"] = filtered_df["Date_Str"].map(date_map)

    # 4. Plotting: Create a 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    sns.set_theme(style="whitegrid")

    # Define the 4 metrics and their titles
    metrics_config = [
        ("ttft_ms", "Time To First Token (ms) - Lower is Better", axes[0, 0]),
        ("itl_ms", "Inter-Token Latency (ms) - Lower is Better", axes[0, 1]),
        ("in_tps", "Input Throughput (tokens/s) - Higher is Better", axes[1, 0]),
        ("out_tps", "Output Throughput (tokens/s) - Higher is Better", axes[1, 1]),
    ]

    for metric, title, ax in metrics_config:
        if metric in filtered_df.columns:
            sns.lineplot(
                data=filtered_df,
                x="x_idx",
                y=metric,
                hue="Legend_Entry",
                marker="o",
                markersize=6,
                linewidth=2,
                ax=ax,
            )
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_ylabel(metric)
            ax.set_xlabel("")

            # Set X-axis labels
            ax.set_xticks(range(len(unique_dates)))
            ax.set_xticklabels(unique_dates, rotation=45, ha="right", fontsize=8)

            # Handle legend: show only on the first plot or make it smaller
            ax.legend(fontsize=8, title=None)
        else:
            ax.text(0.5, 0.5, f"{metric} not found", ha="center", transform=ax.transAxes)

    # 5. Overall layout adjustment
    config_str = f"Config: Concurrency={args.concurrency}, Input={args.input}, Output={args.output}"
    plt.suptitle(f"SGLang JAX CI Performance Trend\n{config_str}", fontsize=16, y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)  # Leave space for suptitle

    plt.savefig(args.output_file, dpi=300)
    print(f"Visualization saved to: {args.output_file}")
    print(f"Total dates included: {len(unique_dates)}")


if __name__ == "__main__":
    cmd_args = get_args()
    http_session = get_session(proxy=cmd_args.proxy, token=cmd_args.token)

    # 1. Get date folders
    dates = get_date_folders(http_session, cmd_args)

    if dates:
        # 2. Fetch data
        combined_df = fetch_data(http_session, dates, cmd_args)

        # 3. Plot
        if combined_df is not None:
            plot_perf_results(combined_df, cmd_args)
        else:
            print(f"Error: Failed to fetch valid data from directories.")
