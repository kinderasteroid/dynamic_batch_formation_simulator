import os
import copy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import the simulator components you created earlier
from src.request_queue import WorkloadGenerator, RequestQueue
from src.engine_wrapper import LLMEngine
from src.metrics import MetricsTracker
from src.schedulers.fixed_batching import FixedBatchScheduler
from src.schedulers.dynamic_batching import DynamicBatchScheduler
from src.schedulers.continuous_batching import ContinuousBatchScheduler

def plot_results(output_dir: str):
    """Reads the generated CSVs and creates the comparison graphs."""
    print("Generating performance graphs...")
    
    # Load data
    df_fixed = pd.read_csv(f"{output_dir}/fixed_metrics.csv")
    df_dynamic = pd.read_csv(f"{output_dir}/dynamic_metrics.csv")
    df_continuous = pd.read_csv(f"{output_dir}/continuous_metrics.csv")

    df_fixed['Strategy'] = 'Fixed Batching'
    df_dynamic['Strategy'] = 'Dynamic Batching'
    df_continuous['Strategy'] = 'Continuous Batching'
    
    df_all = pd.concat([df_fixed, df_dynamic, df_continuous])

    sns.set_theme(style="whitegrid")

    # 1. Average Time To First Token (TTFT)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_all, x='Strategy', y='ttft', errorbar=None)
    plt.title('Average Time To First Token (Lower is Better)')
    plt.ylabel('Seconds')
    plt.savefig(f"{output_dir}/ttft_comparison.png")
    plt.close()

    # 2. System Throughput (Tokens per Second)
    # Throughput = Total Output Tokens / Total Time Taken
    strategies = ['Fixed Batching', 'Dynamic Batching', 'Continuous Batching']
    throughputs = []
    for df in [df_fixed, df_dynamic, df_continuous]:
        total_tokens = df['output_tokens'].sum()
        total_time = df['end_time'].max() - df['arrival_time'].min()
        throughputs.append(total_tokens / total_time)

    plt.figure(figsize=(8, 5))
    sns.barplot(x=strategies, y=throughputs)
    plt.title('Overall System Throughput (Higher is Better)')
    plt.ylabel('Tokens / Second')
    plt.savefig(f"{output_dir}/throughput_comparison.png")
    plt.close()

    # 3. End-to-End Latency CDF (Cumulative Distribution Function)
    plt.figure(figsize=(8, 5))
    sns.ecdfplot(data=df_all, x='e2e_latency', hue='Strategy')
    plt.title('CDF of End-to-End Latency (Left/Steep is Better)')
    plt.xlabel('Latency (Seconds)')
    plt.ylabel('Proportion of Requests Completed')
    plt.savefig(f"{output_dir}/latency_cdf.png")
    plt.close()

    print(f"Graphs successfully saved to {output_dir}/")

def main():
    # Fetch the output directory defined in docker-compose.yml
    output_dir = os.getenv("OUTPUT_DIR", "./results")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating synthetic workload (1,000 requests)...")
    # Generate 1,000 requests arriving roughly every 0.2 seconds (5 req/sec)
    base_requests = WorkloadGenerator.generate_workload(num_requests=1000, arrival_rate=5.0)

    # ==========================================
    # 1. Run Fixed Batching
    # ==========================================
    print("Running Fixed Batching Simulator...")
    engine_f = LLMEngine(max_batch_size=16)
    queue_f = RequestQueue(copy.deepcopy(base_requests))
    metrics_f = MetricsTracker()
    
    FixedBatchScheduler(engine_f, queue_f, metrics_f, batch_size=16).run()
    metrics_f.export_to_csv(f"{output_dir}/fixed_metrics.csv")

    # ==========================================
    # 2. Run Dynamic Batching
    # ==========================================
    print("Running Dynamic Batching Simulator...")
    engine_d = LLMEngine(max_batch_size=16)
    queue_d = RequestQueue(copy.deepcopy(base_requests))
    metrics_d = MetricsTracker()
    
    DynamicBatchScheduler(engine_d, queue_d, metrics_d, max_batch_size=16).run()
    metrics_d.export_to_csv(f"{output_dir}/dynamic_metrics.csv")

    # ==========================================
    # 3. Run Continuous Batching
    # ==========================================
    print("Running Continuous Batching Simulator...")
    engine_c = LLMEngine(max_batch_size=16)
    queue_c = RequestQueue(copy.deepcopy(base_requests))
    metrics_c = MetricsTracker()
    
    ContinuousBatchScheduler(engine_c, queue_c, metrics_c, max_batch_size=16).run()
    metrics_c.export_to_csv(f"{output_dir}/continuous_metrics.csv")

    # ==========================================
    # 4. Generate Output Graphs
    # ==========================================
    plot_results(output_dir)
    print("Experiment complete! Check your local 'results' folder.")

if __name__ == "__main__":
    main()