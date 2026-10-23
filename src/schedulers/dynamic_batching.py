from src.request_queue import RequestQueue
from src.engine_wrapper import LLMEngine
from src.metrics import MetricsTracker, RequestMetric

class DynamicBatchScheduler:
    def __init__(self, engine: LLMEngine, queue: RequestQueue, metrics: MetricsTracker, max_batch_size: int = 8):
        self.engine = engine
        self.queue = queue
        self.metrics = metrics
        self.max_batch_size = max_batch_size

    def run(self):
        while self.queue.has_pending():
            self.queue.update_queue(self.engine.current_time)
            
            if not self.queue.waiting:
                self.engine.current_time = self.queue.incoming[0].arrival_time
                continue

            # Take whatever is available right now, up to max capacity
            batch_size = min(len(self.queue.waiting), self.max_batch_size)
            batch = self.queue.waiting[:batch_size]
            self.queue.waiting = self.queue.waiting[batch_size:]
            
            active_metrics = {}
            for req in batch:
                active_metrics[req.id] = RequestMetric(
                    request_id=req.id, arrival_time=req.arrival_time, start_time=self.engine.current_time
                )

            self.engine.prefill(batch)
            for req in batch:
                active_metrics[req.id].first_token_time = self.engine.current_time

            active_reqs = batch[:]
            while active_reqs:
                completed = self.engine.decode_step(active_reqs)
                for req in completed:
                    metric = active_metrics[req.id]
                    metric.end_time = self.engine.current_time
                    metric.input_tokens = req.input_len
                    metric.output_tokens = req.generated_tokens
                    self.metrics.log_request(metric)
                    active_reqs.remove(req)