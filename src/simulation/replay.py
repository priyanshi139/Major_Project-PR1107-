import threading, time


class DataReplayEngine:
    """Replays HISTORICAL observations in timestamp order. This is a simulation, not a live patient feed.
    Runs in a background thread so a UI never freezes. step() allows synchronous single-stepping."""

    def __init__(self, observations, on_observation, on_reset=None, base_interval=1.0, speed=1.0):
        hours = [o.hour for o in observations]
        if hours != sorted(set(hours)):
            raise ValueError("Observations must have strictly increasing, unique hours")
        self.obs, self.on_observation, self.on_reset = list(observations), on_observation, on_reset
        self.base_interval, self._speed = base_interval, speed
        self._pos, self._status, self.last_error = 0, "idle", None
        self._lock = threading.Lock()
        self._run, self._stop = threading.Event(), threading.Event()
        self._thread = None

    status = property(lambda s: s._status)
    position = property(lambda s: s._pos)
    total = property(lambda s: len(s.obs))
    speed = property(lambda s: s._speed)

    def set_speed(self, speed):
        if speed <= 0:
            raise ValueError("speed must be > 0")
        self._speed = speed

    def step(self):
        """Process exactly one observation. Returns callback result or None if finished."""
        with self._lock:
            if self._pos >= len(self.obs):
                self._status = "finished"
                return None
            o = self.obs[self._pos]
            self._pos += 1
        res = self.on_observation(o)
        if self._pos >= len(self.obs):
            self._status = "finished"
        return res

    def _loop(self):
        while not self._stop.is_set():
            if not self._run.wait(timeout=0.05):
                continue
            if self._stop.is_set() or self._pos >= len(self.obs):
                break
            try:
                self.step()
            except Exception as e:
                self.last_error, self._status = str(e), "error"
                return
            if self._stop.wait(self.base_interval / self._speed):
                break
        if self._status == "running":
            self._status = "finished"

    def start(self):
        if self._status == "paused":
            return self.resume()
        if self._status == "running":
            return
        if self._pos >= len(self.obs):
            raise RuntimeError("Replay finished; call reset() first")
        self._stop.clear(); self._run.set(); self._status = "running"
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def pause(self):
        if self._status == "running":
            self._run.clear(); self._status = "paused"

    def resume(self):
        if self._status == "paused":
            self._run.set(); self._status = "running"

    def stop(self):
        self._stop.set(); self._run.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def reset(self):
        self.stop()
        with self._lock:
            self._pos, self._status, self.last_error = 0, "idle", None
        self._run.clear(); self._stop.clear(); self._thread = None
        if self.on_reset:
            self.on_reset()
