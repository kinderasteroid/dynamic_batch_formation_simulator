import random
from collections import deque
from dataclasses import dataclass
from typing import List

@dataclass
class Request:
    id: int
    arrival_time: float
    input_len: int
    target_output_len: int
    
    # State tracking for the simulator
    generated_tokens: int = 0

class WorkloadGenerator:
    @staticmethod
    def generate_workload(num_requests: int, arrival_rate: float) -> List[Request]:
        """Generates a list of requests with Poisson-like arrivals."""
        requests = []
        current_time = 0.0
        for i in range(num_requests):
            # Simulate varying prompt and output lengths
            input_len = random.randint(10, 200)
            target_output = random.randint(20, 300)
            
            requests.append(Request(
                id=i, 
                arrival_time=current_time, 
                input_len=input_len, 
                target_output_len=target_output
            ))
            # Advance time based on arrival rate (requests per second)
            current_time += random.expovariate(arrival_rate)
            
        return requests

class RequestQueue:
    def __init__(self, requests: List[Request]):
        # Sort by arrival time just in case
        self.incoming = deque(sorted(requests, key=lambda x: x.arrival_time))
        self.waiting: List[Request] = []

    def update_queue(self, current_time: float):
        """Moves requests from incoming to waiting if their arrival time has passed."""
        while self.incoming and self.incoming[0].arrival_time <= current_time:
            self.waiting.append(self.incoming.popleft())

    def has_pending(self) -> bool:
        return len(self.incoming) > 0 or len(self.waiting) > 0