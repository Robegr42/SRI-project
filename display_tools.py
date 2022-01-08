import time


class _ProgressBar:
    def __init__(self, total, width=50):
        self.total = total
        self.width = width
        self.current = 0
        self.started = False
        self.start_time = None

    def update(self, current):
        if not self.started:
            self.started = True
            self.start_time = time.time()
        self.current = current
        self.display()

    def display(self):
        percent = self.current / self.total
        filled_width = int(self.width * percent)
        bar_text = "#" * filled_width + "." * (self.width - filled_width)
        time_passed = time.time() - self.start_time

        percent *= 100

        if self.current == 0:
            time_left = "--:--:--"
        else:
            time_left = (time_passed / self.current) * (self.total - self.current)
            time_left = time.strftime("%H:%M:%S", time.gmtime(time_left))
        time_passed = time.strftime("%H:%M:%S", time.gmtime(time_passed))

        print(f"\r[{bar_text}] {percent:.2f}% {time_passed}  ETA: {time_left}", end="")
        if self.current == self.total:
            print()

    def iter(self, iterable):
        self.update(0)
        for i, item in enumerate(iterable):
            yield item
            self.update(i + 1)


def pbar(iterable):
    pbar = _ProgressBar(len(iterable))
    for item in pbar.iter(iterable):
        yield item
