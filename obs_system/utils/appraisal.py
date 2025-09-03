from obs_system.utils.logger import logger 

import time
import threading
import traceback 

from time import perf_counter_ns 
from collections import defaultdict, deque 
from contextlib import contextmanager 

class PerfMetric(): 

    def __init__(self,frame_budget_ms=200, keep_last=120, use_cuda=False, name=""): 
        self.frame_budget_ms = frame_budget_ms
        self.keep_last = keep_last 
        self.use_cuda = use_cuda 
        self.name = name 
        self.total_ns = defaultdict(int)
        self.calls = defaultdict(int) 
        self.last_frame = {}
        self.history = defaultdict(lambda: deque(maxlen=self.keep_last))
        self._local = threading.local() 
        if use_cuda: 
            import torch 
            self.torch = torch  
        else: 
            self.torch = None  

        self.t0:float = 0.0 
        self.start = None 
        self.end = None 
 
    def __enter__(self):
        if self.torch: 
            self.start = self.torch.cuda.Event(enable_timing=True) 
            self.end = self.torch.cuda.Event(enable_timing=True) 
            self.torch.cuda.synchronize() 
            self.start.record() 
        else: 
            self.t0 = perf_counter_ns() 

        return 

    def __exit__(self, *args): 
        if self.torch: 
            self.end.record() 
            self.torch.cuda.syncrhonize() 
            ms = self.start.elapsed_time(self.end) 
            ns = int(ms*1e6) 
        else: 
            ns = perf_counter_ns() - self.t0 

        self.total_ns[self.name] += ns 
        self.calls[self.name] += 1 
        self.last_frame[self.name] += ns 
        self.history[self.name].append(ns) 

        logger.debug(f"Performance of {self.name} is {ns}")
        return True 

    
    def name_setter(self, name): 
        self.name = name 


    def start_frame(self): 
        self.last_frame.clear() 


    def end_frame(self): 
        total_ns = sum(self.last_frame.values()) 
        total_ms = total_ns / 1e6 
        budget_ok = total_ms <= self.frame_budget_ms
        parts = " | ".join(f"{k}:{v/1e6:.1f}ms" for k,v in self.last_frame.items())
        logger.debug(
                f"[frame] {parts} || total:{total_ms:.1f}ms"
                f"{'✓' if budget_ok else '✗>budget'} (≤{self.frame_budget_ms:.0f}ms)"
        )

    
    def summary(self): 
        logger.debug("=== Performance Results ===") 

        grand_ns = 0 
        for k in sorted(self.total_ns.keys()): 
            total_ns = self.total_ns[k] 
            n = max(self.calls[k],1)
            grand_ns += total_ns 
            avg_ms = total_ns / n / 1e6 
            p_95_ms = self._percentile(self.history,95) / 1e6 if self.history[k] else 0.0
            logger.info(f"{k:>12s}:avg {avg_ms:.2f}ms | p95 {p_95_ms:.2f}ms | calls {n}")

        logger.info(f"{'Grand Total':>12s} : {grand_ns/1e6:.2f} ms across modules") 


    @staticmethod
    def _percentile(values, p): 
        if not values: 
            return 0 

        v = sorted(values) 
        k = ((len(v)-1)*(p/100))
        f = int(k) 
        c = min(f + 1, len(v)-1) 
        if f == c: return v[int(k)]

        return v[f] + (v[c] - v[f]) * (k - f) 


class Timer(): 

    def __init__(self, name): 
        self.name = name 


    def __enter__(self): 
        self.start = time.perf_counter() 
        return self 


    def __exit__(self, *args): 
        self.end = time.perf_counter()
        elapsed = self.end - self.start 
        logger.info(f"[{self.name}] took elapsed time : {elapsed/1e3:.4f}ms")
                    


class SetupError(RuntimeError): 
    def __init__(self, step, exc_value): 
        super().__init__(f"[{step}] failed: {exc_value}")
        self.step = step 
        self.original = exc_value 


class StepContext(): 

    def __init__(self, name, *, catch=(Exception,),supress=False,on_error=None):
        self.name = name 
        self.catch = catch
        self.supress = supress
        self.on_error = on_error 

        self.t0:float = 0.0 
        self.elapsed_time:float = 0.0 
        self.no_exception_found = None 


    def __enter__(self) : 
        logger.debug(f"✅SCM -> {self.name}...")
        self.t0 = time.perf_counter() 
        return self 


    def __exit__(self, exc_type, exc_value, exc_tb): 
        self.elapsed_time = (time.perf_counter() - self.t0) * 1e3
        if exc_value is None: 
            self.no_exception_found =True 
            logger.info(f"[{self.name}] took {self.elapsed_time:.2f}ms")
            return False #Nothing to suppress 

        self.no_exception_found = False 
        if not isinstance(exc_value, self.catch):
            logger.debug(f"Unnexpected Error [{self.name}]: {self.elapsed_time:.2f}ms")
            return False 
        trace_back_str = "".join(traceback.format_exception(exc_type,exc_value,exc_tb))
        if self.on_error: 
            self.on_error(self.name, exc_value, trace_back_str)

        if self.supress: 
            logger.debug(f"[!] [{self.name}] failed but optional:{exc_value}({self.elapsed_time:.2f}ms)")
            return True 
        
        else: 
            raise SetupError(self.name, exc_value)



