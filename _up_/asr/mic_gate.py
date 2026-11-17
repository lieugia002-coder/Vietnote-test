"""Silences microphone audio quieter than a level the user chose.

Someone speaking into the microphone is louder than people talking across the room, so a
threshold between the two keeps only the user. Quiet stretches become silence rather than being
dropped, so word timings still line up with the audio. The gate looks 0.1 s ahead, delaying the
audio by as much, so the soft start of a word is kept, and stays open 0.3 s after the voice
drops, so the end of a word is kept too.
"""
import numpy as np

RATE = 16000
FRAME = RATE // 50                 # 20 ms
LOOKAHEAD = RATE // 10             # 0.1 s
HOLD = RATE * 3 // 10              # 0.3 s
LOWEST_DB, HIGHEST_DB = -70.0, -10.0


class MicGate:
    def __init__(self):
        self.threshold = None      # linear peak, or None when off
        self.reset()

    def reset(self):
        self.pending = None        # the last LOOKAHEAD samples, not yet passed on
        self.offset = 0            # sample index of pending[0]
        self.open_until = 0        # sample index the gate stays open until

    def configure(self, threshold_db):
        """A level in dBFS (peak), or None to let everything through."""
        if threshold_db is None:
            self.threshold = None
            return
        value = min(max(float(threshold_db), LOWEST_DB), HIGHEST_DB)
        self.threshold = 10 ** (value / 20)

    def process(self, audio, captured_at):
        """Returns (audio, captured_at) to pass on, as long as the input; delayed while gating."""
        audio = np.asarray(audio, dtype=np.float32)
        if self.threshold is None:
            if self.pending is None:
                return audio, captured_at
            # Turned off: hand over what was held back, then pass audio straight through.
            out = np.concatenate([self.pending, audio])
            self.reset()
            return out, captured_at
        if self.pending is None:
            self.pending = np.zeros(LOOKAHEAD, dtype=np.float32)
        buffer = np.concatenate([self.pending, audio])
        mask = np.zeros(len(buffer), dtype=bool)
        mask[:max(0, min(len(buffer), self.open_until - self.offset))] = True
        for start in range(0, len(buffer), FRAME):
            if np.abs(buffer[start:start + FRAME]).max() >= self.threshold:
                end = start + FRAME + HOLD
                mask[max(0, start - LOOKAHEAD):end] = True
                self.open_until = max(self.open_until, self.offset + end)
        out = np.where(mask[:len(audio)], buffer[:len(audio)], 0.0).astype(np.float32)
        self.pending = buffer[len(audio):]
        self.offset += len(audio)
        return out, captured_at - LOOKAHEAD / RATE
