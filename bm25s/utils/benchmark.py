from copy import deepcopy
import time
import sys

try:
    import resource
except ImportError:
    print("resource module not available on Windows")
    resource = None


def get_max_memory_usage(format="GB"):
    if resource is None:
        return None
    if format not in ["GB", "MB", "KB"]:
        raise ValueError("format should be one of 'GB', 'MB', 'KB'")

    usage_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # for mac, ru_maxrss is in bytes
    
    if sys.platform == "darwin":
        usage_kb /= 1024
    
    if format == "GB":
        return usage_kb / (1024**2)
    elif format == "MB":
        return usage_kb / 1024
    else:
        return usage_kb


class Timer:
    def __init__(self, prefix="", precision=4):
        self.results = {}
        self.prefix = prefix
        self.precision = precision

    def start(self, name):
        if name in self.results:
            raise ValueError(f"Timer with name {name} already started.")
        start_time = time.monotonic()
        self.results[name] = {"start": start_time, "elapsed": 0, "last": start_time}
        return name

    def stop(self, name, show=False, n_total=None):
        if name not in self.results:
            raise ValueError(f"Timer with name {name} not started.")

        stop_time = time.monotonic()
        r = self.results[name]
        r["stopped"] = stop_time
        r["elapsed"] += stop_time - r.pop("last")

        if show:
            self.show(name, n_total=n_total)

        return self.results[name]["elapsed"]

    def pause(self, name):
        # if self.has_stopped(name):
        #     raise ValueError(f"Timer with name {name} already stopped.")

        # if not self.has_started(name):
        #     raise ValueError(f"Timer with name {name} not started.")

        paused_time = time.monotonic()
        r = self.results[name]

        r["elapsed"] += paused_time - r["last"]

    def resume(self, name):
        # if not self.has_started(name):
        #     raise ValueError(f"Timer with name {name} not started.")

        # if not self.is_paused(name):
        #     raise ValueError(f"Timer with name {name} not paused.")

        # if self.has_stopped(name):
        #     raise ValueError(f"Timer with name {name} already stopped.")

        self.results[name]["last"] = time.monotonic()

    def is_paused(self, name):
        return name in self.results and "paused" in self.results[name]

    def is_resumed(self, name):
        return name in self.results and "resumed" in self.results[name]

    def has_started(self, name):
        return name in self.results

    def has_stopped(self, name):
        return self.has_started(name) and "stopped" in self.results[name]

    def elapsed(self, name, precision=None):
        if precision is None:
            precision = self.precision

        if not self.has_started(name):
            raise ValueError(f"Timer with name {name} not started.")
        if not self.has_stopped(name):
            raise ValueError(f"Timer with name {name} not stopped.")

        return round(self.results[name]["elapsed"], precision)

    def show(self, name, offset=0, n_total=None):
        t = self.elapsed(name) + offset
        s = f"{self.prefix} {name}: {t:.4f}s"
        if n_total is not None:
            # calculate throughput
            throughput = n_total / t
            s += f" ({throughput:.2f}/s)"
        print(s)

    def show_all(self):
        for name in self.results:
            if self.has_stopped(name):
                self.show(name)

    def to_dict(self, underscore=False, lowercase=False):
        results_to_save = deepcopy(self.results)
        if underscore:
            results_to_save = {
                k.replace(" ", "_"): v for k, v in results_to_save.items()
            }

        if lowercase:
            results_to_save = {k.lower(): v for k, v in results_to_save.items()}

        return results_to_save
