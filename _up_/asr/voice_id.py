"""Names speakers from voices the user enrolled, on this device.

The recognizer already splits a meeting into speakers ("Người nói 2"). This module answers who
each of them is: it gathers a speaker's audio, turns it into a voice embedding (CAM++, 192
values) and compares it with the enrolled profiles. A whole speaker is named at once, never a
single line, because a 1-2 s line gives an unsteady embedding while ~8 s is reliable.

Measured on Vietnamese speech through a video-call codec (Opus 16 kbps), with the user's
profiles taken from clean audio: an 8 s group scores 0.85 against its own profile (0.81 at the
10th percentile) and a stranger's best score is 0.55 (0.62 at the 90th). At 0.70 every enrolled
speaker was named and no stranger was; at 0.60, 19% of strangers were given someone's name.
A 3 s group is named at 0.70 nine times in ten, so names show early and settle by 8 s.
"""
import math
import queue
import threading

import numpy as np

RATE = 16000
THRESHOLD = 0.70
# A speaker gets a provisional name once this much of their speech is in, a settled one later.
PROVISIONAL_SECONDS = 3.0
SETTLED_SECONDS = 8.0
# Lines shorter than this add little and often carry another speaker's edge.
MIN_PIECE_SECONDS = 1.0
KEEP_SECONDS = 30.0
MIN_ENROLL_SECONDS = 3.0


# --- Features: Kaldi-style 80-bin log-mel filterbank, the input the embedding model expects.

FRAME = 400            # 25 ms
SHIFT = 160            # 10 ms
FFT = 512
BINS = 80
LOW_HZ, HIGH_HZ = 20.0, RATE / 2 - 400.0


def _mel(hz):
    return 1127.0 * np.log(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def _mel_banks():
    low, high = _mel(LOW_HZ), _mel(HIGH_HZ)
    delta = (high - low) / (BINS + 1)
    mel = _mel(np.arange(FFT // 2) * RATE / FFT)
    banks = np.zeros((BINS, FFT // 2 + 1), dtype=np.float32)
    for b in range(BINS):
        left, center, right = low + b * delta, low + (b + 1) * delta, low + (b + 2) * delta
        up = (mel - left) / (center - left)
        down = (right - mel) / (right - center)
        weight = np.where((mel > left) & (mel < right), np.where(mel <= center, up, down), 0.0)
        banks[b, :FFT // 2] = weight
    return banks


_BANKS = _mel_banks()
_WINDOW = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(FRAME) / (FRAME - 1))) ** 0.85


def fbank(audio):
    """Log-mel features (frames x 80) as Kaldi computes them with snip_edges off and no dither."""
    x = np.asarray(audio, dtype=np.float32)
    n = len(x)
    frames = (n + SHIFT // 2) // SHIFT
    if frames < 1:
        return np.zeros((0, BINS), dtype=np.float32)
    starts = np.arange(frames) * SHIFT + SHIFT // 2 - FRAME // 2
    index = starts[:, None] + np.arange(FRAME)[None, :]
    # Samples before the start or past the end are mirrored, as Kaldi does.
    index = np.where(index < 0, -index - 1, index)
    index = np.where(index >= n, 2 * n - 1 - index, index)
    index = np.clip(index, 0, n - 1)
    f = x[index].astype(np.float64)
    f -= f.mean(axis=1, keepdims=True)
    f[:, 1:] -= 0.97 * f[:, :-1]
    f[:, 0] -= 0.97 * f[:, 0]
    f *= _WINDOW
    power = np.abs(np.fft.rfft(f, n=FFT, axis=1)) ** 2
    energies = power.astype(np.float32) @ _BANKS.T
    return np.log(np.maximum(energies, np.finfo(np.float32).eps)).astype(np.float32)


class Embedder:
    """CAM++ speaker embeddings through onnxruntime; loaded once, used from one thread."""
    def __init__(self, model_path):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), options, providers=['CPUExecutionProvider'])
        self.input = self.session.get_inputs()[0].name

    def embed(self, audio):
        features = fbank(audio)
        if len(features) < 10:
            raise ValueError('audio too short')
        features = features - features.mean(axis=0, keepdims=True)
        vector = self.session.run(None, {self.input: features[None]})[0][0].astype(np.float32)
        return vector / (np.linalg.norm(vector) + 1e-9)


def available():
    try:
        import onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


class _Group:
    def __init__(self):
        self.pieces = []          # float32 arrays of this speaker's lines, newest last
        self.seconds = 0.0        # all speech seen, including what was dropped from pieces
        self.checked_at = 0.0     # seconds when last compared
        self.name = None
        self.score = None
        self.settled = False
        self.announced = (None, False)

    def add(self, audio):
        self.pieces.append(audio)
        self.seconds += len(audio) / RATE
        kept = sum(len(p) for p in self.pieces)
        while len(self.pieces) > 1 and kept - len(self.pieces[0]) >= KEEP_SECONDS * RATE:
            kept -= len(self.pieces.pop(0))

    def audio(self):
        return np.concatenate(self.pieces) if self.pieces else np.zeros(0, dtype=np.float32)


class VoiceNamer:
    """Gathers each speaker's audio and names them from the enrolled profiles.

    `send(label, name, score, settled)` is called from the namer's own thread whenever a
    speaker's name changes. Embeddings run on that thread so recognition never waits on them.
    """
    def __init__(self, send=None):
        self.send = send
        self.lock = threading.Lock()
        self.embedder = None
        self.model_path = None
        self.profiles = {}        # name -> unit vector
        self.threshold = THRESHOLD
        self.groups = {}          # speaker label -> _Group
        self.manual = {}          # label -> name the user typed; never overridden
        self.jobs = queue.Queue()
        self.thread = threading.Thread(target=self.run, daemon=True, name='voice-id')
        self.thread.start()

    # Setup ----------------------------------------------------------------

    def configure(self, model_path, profiles, threshold=None):
        """Profiles: [{'name': str, 'embedding': [192 floats]}]. An empty model path turns naming off."""
        cleaned = {}
        for profile in profiles or []:
            name = str(profile.get('name', '')).strip()
            vector = np.asarray(profile.get('embedding') or [], dtype=np.float32)
            if name and vector.ndim == 1 and vector.size and np.isfinite(vector).all():
                cleaned[name] = vector / (np.linalg.norm(vector) + 1e-9)
        with self.lock:
            self.profiles = cleaned
            if threshold is not None and 0.3 <= float(threshold) <= 0.95:
                self.threshold = float(threshold)
            if model_path != self.model_path:
                self.model_path = model_path
                self.embedder = None
            for group in self.groups.values():
                group.checked_at = 0.0
                group.settled = False
        self.jobs.put(('recheck', None))

    @property
    def active(self):
        return bool(self.model_path)

    def reset(self):
        with self.lock:
            self.groups = {}
            self.manual = {}

    # Feeding ----------------------------------------------------------------

    def observe(self, label, audio):
        """One recognized line of `label`'s speech (float32, 16 kHz)."""
        if not self.active or len(audio) < MIN_PIECE_SECONDS * RATE:
            return
        with self.lock:
            group = self.groups.setdefault(label, _Group())
            group.add(np.asarray(audio, dtype=np.float32))
            due = self._due(group)
        if due:
            self.jobs.put(('check', label))

    def _due(self, group):
        if not self.profiles or group.seconds < PROVISIONAL_SECONDS:
            return False
        if group.checked_at == 0.0:
            return True
        if not group.settled and group.seconds >= SETTLED_SECONDS:
            return True
        # Look again as more speech comes in; a long meeting keeps confirming the name.
        return group.seconds - group.checked_at >= 2 * SETTLED_SECONDS

    # Manual names -------------------------------------------------------------

    def set_name(self, label, name):
        with self.lock:
            if name:
                self.manual[label] = name
            else:
                self.manual.pop(label, None)

    def group_seconds(self, label):
        with self.lock:
            group = self.groups.get(label)
            return group.seconds if group else 0.0

    def embed_group(self, label):
        """The embedding of everything kept for `label`, to save as a profile."""
        with self.lock:
            group = self.groups.get(label)
            audio = group.audio() if group else np.zeros(0, dtype=np.float32)
        if len(audio) < MIN_ENROLL_SECONDS * RATE:
            raise ValueError('not_enough_speech')
        return self._embedder().embed(audio)

    def embed(self, audio):
        if len(audio) < MIN_ENROLL_SECONDS * RATE:
            raise ValueError('not_enough_speech')
        return self._embedder().embed(audio)

    # Work --------------------------------------------------------------------

    def _embedder(self):
        with self.lock:
            path, embedder = self.model_path, self.embedder
        if not path:
            raise RuntimeError('model_missing')
        if embedder is None:
            embedder = Embedder(path)
            with self.lock:
                if self.model_path == path:
                    self.embedder = embedder
        return embedder

    def run(self):
        while True:
            kind, label = self.jobs.get()
            if kind == 'stop':
                return
            try:
                labels = [label] if kind == 'check' else list(self.groups)
                for item in labels:
                    self._check(item)
            except Exception as exc:  # a broken model must not stop recognition
                import sys
                print(f'[VOICE-ID] {exc}', file=sys.stderr, flush=True)

    def _check(self, label):
        with self.lock:
            group = self.groups.get(label)
            if group is None or not self.profiles or group.seconds < PROVISIONAL_SECONDS:
                return
            audio, seconds = group.audio(), group.seconds
            names = list(self.profiles)
            matrix = np.stack([self.profiles[n] for n in names])
        vector = self._embedder().embed(audio)
        scores = matrix @ vector
        best = int(np.argmax(scores))
        score = round(float(scores[best]), 2)
        import sys
        ranked = ', '.join(f'{names[i]} {scores[i]:.2f}' for i in np.argsort(-scores)[:3])
        print(f'[VOICE-ID] {label} · {seconds:.1f} s: {ranked}', file=sys.stderr, flush=True)
        events = []
        with self.lock:
            if self.groups.get(label) is not group:
                return
            group.checked_at = seconds
            group.settled = seconds >= SETTLED_SECONDS
            name = names[best] if score >= self.threshold else None
            # One name per speaker. A name the user typed, or one held by a closer match, stays put.
            holder = None
            if name:
                if name in self.manual.values():
                    name = None
                else:
                    holder = next((l for l, g in self.groups.items() if l != label and g.name == name), None)
                    if holder is not None and (self.groups[holder].score or 0) >= score:
                        name, holder = None, None
            if holder is not None:
                taken = self.groups[holder]
                taken.name, taken.score, taken.announced = None, None, (None, False)
                events.append((holder, None, None, False))
            group.name, group.score = name, score
            state = (name, bool(name) and group.settled)
            if state != group.announced:
                group.announced = state
                events.append((label, name, score, group.settled))
        for event in events:
            self._announce(*event)

    def _announce(self, label, name, score, settled):
        if self.send and label not in self.manual:
            try:
                self.send(label, name, score, settled)
            except OSError:
                pass

    def close(self):
        # Let a running embedding finish: the runtime fails if the process exits mid-inference.
        self.jobs.put(('stop', None))
        self.thread.join(timeout=3)
