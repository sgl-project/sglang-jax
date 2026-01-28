import argparse
import sys
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
        description="SGLang JAX CI Performance Trend Visualization Tool"
    )

    # --- Basic Configuration ---
    parser.add_argument(
        "--owner", type=str, default="pathfinder-pf", help="GitHub repository owner"
    )
    parser.add_argument(
        "--repo", type=str, default="sglang-jax-ci-data", help="GitHub repository name"
    )
    parser.add_argument(
        "--subdir", type=str, default="benchmark", help="Subdirectory containing the data"
    )
    parser.add_argument(
        "--file",
        type=str,
        default="bailing_moe_benchmark_tp_2_ep_2_results.csv",
        help="Target CSV filename",
    )
    parser.add_argument(
        "--metric", type=str, default="mmlu_pro", help="Metric column name to analyze"
    )

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
        "--output", type=str, default="accuracy_trend.png", help="Output image filename"
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
        # Get all directories and sort them (assuming directory names are date formats, e.g., 2024-05-20)
        all_dates = sorted([item["name"] for item in response.json() if item["type"] == "dir"])

        # --- Core Logic: If limit is set, take the last n items ---
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


def plot_results(df, args):
    if df is None or df.empty:
        print("No data available to plot")
        return

    metric = args.metric
    df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df = df.dropna(subset=[metric]).sort_values("Date_Str")
    df["Legend_Entry"] = df["Model"] + " (" + metric + ")"

    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")

    unique_dates = df["Date_Str"].unique()
    date_map = {date: i for i, date in enumerate(unique_dates)}
    df["x_idx"] = df["Date_Str"].map(date_map)

    sns.lineplot(
        data=df, x="x_idx", y=metric, hue="Legend_Entry", marker="o", markersize=8, linewidth=2
    )

    plt.title(
        f"SGLang JAX CI: {metric} Trend (Latest {len(unique_dates)} points)", fontsize=14, pad=15
    )
    plt.ylabel("Score", fontsize=12)
    plt.xlabel("Date", fontsize=12)

    plt.xticks(
        ticks=range(len(unique_dates)), labels=unique_dates, rotation=45, fontsize=8, ha="right"
    )

    plt.legend(loc="upper left", fontsize=9, title="Model & Metric", shadow=True)
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Visualization saved to: {args.output}")


if __name__ == "__main__":
    cmd_args = get_args()
    http_session = get_session(proxy=cmd_args.proxy, token=cmd_args.token)

    # 1. Get (and filter) date folders
    dates = get_date_folders(http_session, cmd_args)

    if dates:
        # 2. Fetch data
        combined_df = fetch_data(http_session, dates, cmd_args)

        # 3. Plot
        if combined_df is not None:
            plot_results(combined_df, cmd_args)
        else:
            print(f"Error: Failed to fetch valid data from directories.")
