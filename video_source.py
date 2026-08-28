import time
import cv2


class RobustVideoCapture:
    """
    Thin wrapper around cv2.VideoCapture that automatically attempts
    to reconnect when the underlying stream drops.

    This matters for config.SOURCE
    (https://cctvn.freeway.gov.tw/abs2mjpg/bmjpg?camera=15771): it's
    an MJPEG-over-HTTP feed, and those regularly hiccup -- brief
    network errors, the camera restarting, gov.tw side hiccups, etc.
    A plain cv2.VideoCapture just starts returning
    (False, None) forever once that happens, which silently kills
    the processing loop. This class detects that and retries with
    exponential backoff instead of giving up on the first failure.

    For a local file (config.LOCAL), a failed read just means the
    video has ended -- there's nothing to "reconnect" to, and
    retrying forever would turn a normal end-of-video into an
    infinite loop. Pass reconnect=False for that case so read()
    simply reports the failure once and stops.
    """

    def __init__(
        self,
        source,
        reconnect=True,
        max_retries=None,
        retry_delay=2.0,
        backoff_factor=1.5,
        max_retry_delay=30.0,
    ):
        self.source = source
        self.reconnect = reconnect
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor
        self.max_retry_delay = max_retry_delay

        self.cap = None
        self._connect()

    def _connect(self):
        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(self.source)
        return self.cap.isOpened()

    def isOpened(self):
        return self.cap is not None and self.cap.isOpened()

    def get(self, prop_id):
        return self.cap.get(prop_id) if self.cap is not None else 0

    def read(self):
        if self.cap is None or not self.cap.isOpened():
            if not self.reconnect or not self._reconnect():
                return False, None

        success, frame = self.cap.read()

        if not success or frame is None:
            if not self.reconnect:
                # Local file (or reconnection disabled): treat this
                # as a normal end-of-stream, not something to retry.
                return False, None

            print("[stream] Read failed, attempting to reconnect...")
            if not self._reconnect():
                return False, None
            success, frame = self.cap.read()

        return success, frame

    def _reconnect(self):
        attempt = 0
        delay = self.retry_delay

        while self.max_retries is None or attempt < self.max_retries:
            attempt += 1

            limit = f"/{self.max_retries}" if self.max_retries else ""
            print(
                f"[stream] Reconnect attempt {attempt}{limit} "
                f"to {self.source} ..."
            )

            if self._connect():
                print("[stream] Reconnected successfully.")
                return True

            time.sleep(delay)
            delay = min(delay * self.backoff_factor, self.max_retry_delay)

        print("[stream] Giving up reconnecting.")
        return False

    def release(self):
        if self.cap is not None:
            self.cap.release()
