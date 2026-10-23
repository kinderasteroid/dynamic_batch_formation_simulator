import csv
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class RequestMetric:
    request_id: int
    arrival_time: float
    start_time: Optional[float] = None
    first_token_time: Optional[float] = None
    end_time: Optional[float] = None
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def ttft(self) -> float:
        """Time To First Token"""
        if self.first_token_time and self.arrival_time:
            return self.first_token_time - self.arrival_time
        return 0.0

    @property
    def e2e_latency(self) -> float:
        """End-to-End Latency"""
        if self.end_time and self.arrival_time:
            return self.end_time - self.arrival_time
        return 0.0

class MetricsTracker:
    def __init__(self):
        self.records: List[RequestMetric] = []

    def log_request(self, metric: RequestMetric):
        self.records.append(metric)

    def export_to_csv(self, filepath: str):
        with open(filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "request_id", "arrival_time", "start_time", "first_token_time", 
                "end_time", "input_tokens", "output_tokens", "ttft", "e2e_latency"
            ])
            for r in self.records:
                writer.writerow([
                    r.request_id, r.arrival_time, r.start_time, r.first_token_time,
                    r.end_time, r.input_tokens, r.output_tokens, r.ttft, r.e2e_latency
                ])