"""Compresses the live audio sent for recognition.

Raw 16 kHz PCM is 256 kbps per source, half a megabit for a meeting's microphone and
computer sound together; on a 3 Mbps uplink shared with a video call the 5 s backlog
filled and old speech was skipped. Ogg/Opus at 32 kbps is about a seventh of that and
recognizes as well. Without PyAV, or once encoding fails, audio goes out as raw PCM.
"""
import io
import os

import numpy as np

BITRATE = 32000
# Bytes leave once a page fills: 60 ms adds that much delay but keeps the per-page
# headers (27 bytes each) from costing 10 kbps the way 20 ms pages do.
PAGE_MS = 60
# Silence that pushes the last spoken page out before a finalize; covers a full page
# plus the encoder's look-ahead.
TAIL_PAD_MS = 160

_failed = False


def log(message):
    print(message, flush=True)


class _Sink(io.RawIOBase):
    def __init__(self):
        self.chunks = []

    def writable(self):
        return True

    def write(self, data):
        self.chunks.append(bytes(data))
        return len(data)

    def take(self):
        data, self.chunks = b''.join(self.chunks), []
        return data


class OpusStream:
    """One Ogg/Opus stream, from its headers on: start a new one for each connection."""
    def __init__(self, rate):
        import av
        self.av = av
        self.rate = rate
        self.sink = _Sink()
        self.container = av.open(self.sink, 'w', format='ogg', options={'page_duration': str(PAGE_MS * 1000)})
        self.stream = self.container.add_stream('libopus', rate=rate, layout='mono')
        self.stream.bit_rate = BITRATE
        # 'audio' rather than 'voip': the computer's sound is often music under speech.
        self.stream.codec_context.options = {'application': 'audio', 'frame_duration': '20', 'vbr': 'on'}
        self.pts = 0

    def encode(self, pcm):
        """Little-endian 16-bit mono PCM in; whatever Ogg bytes are ready out (often none)."""
        global _failed
        try:
            samples = np.frombuffer(pcm, dtype='<i2')
            frame = self.av.AudioFrame.from_ndarray(samples[None, :], format='s16', layout='mono')
            frame.sample_rate, frame.pts = self.rate, self.pts
            self.pts += len(samples)
            for packet in self.stream.encode(frame):
                self.container.mux(packet)
            return self.sink.take()
        except Exception:
            _failed = True  # The next connection sends raw PCM instead of failing again.
            raise

    def drain(self):
        """Pushes out the audio still held for the current page."""
        return self.encode(bytes(2 * self.rate * TAIL_PAD_MS // 1000))


def check():
    """For release builds: True when the frozen worker can compress a second of audio."""
    encoder = opus_stream(16000)
    if encoder is None:
        return False
    pcm = (np.sin(np.arange(16000) / 5) * 8000).astype('<i2').tobytes()
    data = b''.join(encoder.encode(pcm[i:i + 640]) for i in range(0, len(pcm), 640)) + encoder.drain()
    log(f'[ASR] Opus check: {len(data)} bytes for 1 s')
    return data.startswith(b'OggS') and 1000 < len(data) < 10000


def opus_stream(rate):
    """A new encoder, or None to send raw PCM (turned off, unavailable, or failed before)."""
    global _failed
    if _failed or os.environ.get('VIETNOTE_ASR_AUDIO') == 'pcm':
        return None
    try:
        return OpusStream(rate)
    except Exception as exc:
        _failed = True
        log(f'[ASR] compression unavailable, sending raw audio: {exc}')
        return None
