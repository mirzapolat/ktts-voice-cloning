"""Local web UI for KittenTTS.

Run:
    .venv/bin/python app.py
then open http://127.0.0.1:7860
"""
import copy
import functools
import json
import time
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download

from kittenml import KittenTTS
from kittenml.kittentts2 import text as text_utils
from kittenml.kittentts2.model import CHARS_PER_SECOND, TOKENS_PER_SECOND

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

VOICES = {
    "English": [
        "Bella", "Jasper", "Luna", "Bruno", "Rosie", "Hugo", "Kiki", "Leo",
        "Matthew", "Elliot", "Willow", "Dolores", "Victor", "Dante", "Alfred",
        "Saoirse", "Claire", "Raven", "Marcus", "Herbert", "Diana", "Laurence",
        "Maeve", "Walter", "Edith", "Miles", "Grace", "Reginald", "Iris", "Frank",
        "Serena", "Julian", "Eleanor", "Otis", "Vincent", "Martha", "Sable", "Victoria",
    ],
    "Other languages": [
        "Arabic", "Chinese", "French", "German", "Hindi", "Italian",
        "Portuguese", "Russian", "Spanish",
    ],
}
LANGUAGE_VOICES = set(VOICES["Other languages"])
ALL_VOICES = VOICES["English"] + VOICES["Other languages"]

EMOTIONS = ["angry", "contemplative", "excited", "joyful", "mundane",
            "nervous", "sad", "stern", "surprised", "tender"]
EVENTS = ["gasp", "giggle", "growl", "gulp", "laugh", "pause",
          "scoff", "sigh", "sob", "um"]
PRESETS = {
    "stable": dict(temperature=0.8, top_p=0.8, top_k=50, min_p=0.0),
    "expressive": dict(temperature=0.9, top_p=0.9, top_k=50, min_p=0.0),
}

ONNX_MODELS = {
    "mini (80M, best)": "KittenML/kitten-tts-mini-0.8",
    "micro (40M)": "KittenML/kitten-tts-micro-0.8",
    "nano (15M)": "KittenML/kitten-tts-nano-0.8-fp32",
    "nano int8 (15M, smallest)": "KittenML/kitten-tts-nano-0.8-int8",
}
ONNX_VOICES = ["Bella", "Jasper", "Luna", "Bruno", "Rosie", "Hugo", "Kiki", "Leo"]
K2_ENGINE, ONNX_ENGINE = "KittenTTS 2", "Lightweight (ONNX)"

_models = {}


def get_model(repo, **kwargs):
    key = (repo, tuple(sorted(kwargs.items())))
    if key not in _models:
        m = KittenTTS(repo, **kwargs)
        if hasattr(m, "_generate_tokens"):   # KittenTTS 2, not the ONNX models
            m.prefix_cache = PrefixCache(m)
        _models[key] = m
    return _models[key]


def save(audio, sr, tag):
    path = OUT_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{tag}.wav"
    sf.write(path, audio, sr)
    return str(path)


def fmt_secs(s):
    s = max(0, int(round(s)))
    return f"{s // 60}m {s % 60:02d}s" if s >= 60 else f"{s}s"


# ---------------------------------------------------------------- voice prompt cache

class PrefixCache:
    """Reads each voice's part of the prompt once and reuses it.

    Every chunk's prompt starts with the same voice part (reference transcript +
    reference audio, ~300 tokens, ~8 s on a CPU) followed by the chunk's own text.
    The model's attention cache after that voice part is kept, so later chunks and
    later generations with the same voice (built-in or cloned) only read their text.
    """

    MAX_VOICES = 4      # ~60 MB of attention cache each
    MIN_TOKENS = 32     # shorter prefixes aren't worth caching

    def __init__(self, m):
        self.m = m
        self.entries = OrderedDict()   # key -> (embedding, cache); embedding pins the id
        self.reference = None          # voice of the chunk being generated
        self.last_uncached = None      # prompt tokens the last generate() had to read
        self._generate_tokens = m._generate_tokens
        self._generate = m.model.generate
        m._generate_tokens = self.generate_tokens
        m.model.generate = self.generate

    @staticmethod
    def key(reference):
        # references (and their embedding tensors) are cached by the model per voice/clip
        return tuple(reference["prefix"]), id(reference["embedding"])

    def is_warm(self, reference, use_reference_prompt):
        return not use_reference_prompt or self.key(reference) in self.entries

    def generate_tokens(self, reference, chunk, settings, max_new_tokens,
                        use_reference_prompt, *args, **kwargs):
        self.reference = reference if use_reference_prompt else None
        try:
            return self._generate_tokens(reference, chunk, settings, max_new_tokens,
                                         use_reference_prompt, *args, **kwargs)
        finally:
            self.reference = None

    def generate(self, input_ids, **kwargs):
        ref = self.reference
        head = list(ref["prefix"]) if ref else []
        n = len(head)
        if (n < self.MIN_TOKENS or input_ids.shape[1] <= n or "past_key_values" in kwargs
                or input_ids[0, :n].tolist() != head):
            self.last_uncached = input_ids.shape[1]
            return self._generate(input_ids, **kwargs)

        key = self.key(ref)
        if key in self.entries:
            self.entries.move_to_end(key)
            cache = self.entries[key][1]
            self.last_uncached = input_ids.shape[1] - n
        else:
            # the speaker hook injects the voice embedding at position 0 on this pass
            with torch.no_grad():
                cache = self.m.model(input_ids=input_ids[:, :n], use_cache=True).past_key_values
            self.entries[key] = (ref["embedding"], cache)
            while len(self.entries) > self.MAX_VOICES:
                self.entries.popitem(last=False)
            self.last_uncached = input_ids.shape[1]
        # generate() extends the cache in place, so hand it a copy
        return self._generate(input_ids, past_key_values=copy.deepcopy(cache), **kwargs)


# ---------------------------------------------------------------- progress

# Timing model for the ETA, refined after every run. Seeds were measured on an
# Apple Silicon CPU: before its first token each chunk reads the uncached part of
# its prompt, then pays a steady cost per generated token (decoding included).
_learned = dict(
    sec_per_token=0.077,          # LM step + share of audio decoding
    prefill_per_token=0.028,      # reading prompt tokens
    tokens_per_char=1.6,          # how much speech a character of text turns into
)
TOKENS_FLOOR = 30                 # lead-in/out silence every chunk produces regardless of length


def split_chunks(m, text, normalize, advanced):
    spoken = text_utils.normalize_text(text, normalize=normalize)
    return text_utils.split_for_synthesis(
        spoken,
        max_chars=advanced.get("chunk_chars") or m.chunk_chars,
        min_chars=advanced.get("chunk_min_chars") or m.chunk_min_chars) or [spoken]


class ProgressTracker:
    """Drives the progress bar from the model's per-token generation.

    Progress is elapsed / (elapsed + estimated remaining), so prompt reading and
    token generation both move the bar. Estimates switch from the learned values
    to this run's own measurements as soon as they exist.
    """

    def __init__(self, progress, chunks, cache=None):
        self.progress = progress
        self.cache = cache
        self.chars = [len(c) for c in chunks]
        self.n = len(chunks)
        self.head_tokens = 0        # voice part of the prompt; set once the voice is known
        self.warm = True            # whether that voice part is already cached
        self.i = 0                  # index of the chunk being worked on
        self.phase = "voice"        # voice → prefill → generating → decoding → (next chunk)
        self.finished_tokens = []   # tokens of finished chunks
        self.current = 0            # tokens so far in the current chunk
        self.t_start = time.time()
        self.t_chunk = None
        self.t_first_token = None
        self.prefill_time = 0.0     # measured prompt reading, all chunks
        self.prefill_tokens = 0
        self.gen_time = 0.0         # measured first-token → decoded time, all chunks
        self.last_update = 0.0
        self._prompt_seen = False

    def set_voice(self, head_tokens, warm):
        self.head_tokens, self.warm = head_tokens, warm

    def expected_tokens(self, i):
        return TOKENS_FLOOR + self.chars[i] * _learned["tokens_per_char"]

    def prompt_tokens(self, i):
        """Uncached prompt tokens chunk i will read: its text, plus the voice if cold."""
        text = self.chars[i] / 4 + 8
        return text + (self.head_tokens if i == 0 and not self.warm else 0)

    def sec_per_token(self):
        done = sum(self.finished_tokens)
        if done > 20 and self.gen_time > 0:
            return self.gen_time / done
        if self.current > 20 and self.t_first_token:
            return (time.time() - self.t_first_token) / self.current
        return _learned["sec_per_token"]

    def sec_per_prompt_token(self):
        if self.prefill_tokens > 50:
            return self.prefill_time / self.prefill_tokens
        return _learned["prefill_per_token"]

    def remaining(self):
        now = time.time()
        rate, prate = self.sec_per_token(), self.sec_per_prompt_token()
        later = sum(self.prompt_tokens(j) * prate + self.expected_tokens(j) * rate
                    for j in range(self.i + 1, self.n))
        if self.i >= self.n:
            return later
        expected = self.expected_tokens(self.i)
        prefill = self.prompt_tokens(self.i) * prate
        if self.phase == "voice":
            here = prefill + expected * rate
        elif self.phase == "prefill":
            here = max(0.3, prefill - (now - self.t_chunk)) + expected * rate
        elif self.phase == "generating":
            # past the guess: assume it's nearly done rather than claiming 0s
            here = max(expected - self.current, 0.1 * self.current) * rate
        else:  # decoding
            here = 0.0
        return here + later

    def report(self, stage, force=False):
        now = time.time()
        if not force and now - self.last_update < 0.25:
            return
        self.last_update = now
        elapsed, left = now - self.t_start, self.remaining()
        audio = (sum(self.finished_tokens) + self.current) / TOKENS_PER_SECOND
        parts = [f"Chunk {min(self.i + 1, self.n)}/{self.n}", stage]
        if audio:
            parts.append(f"{audio:.1f}s of audio")
        parts.append(f"~{fmt_secs(left)} left")
        self.progress(min(elapsed / (elapsed + left), 0.99), desc=" · ".join(parts))

    # transformers streamer interface: first put() is the prompt, then one per token
    def put(self, value):
        if not self._prompt_seen:
            self._prompt_seen = True
            return
        if self.current == 0:
            self.t_first_token = time.time()
            self.prefill_time += self.t_first_token - self.t_chunk
            self.prefill_tokens += (self.cache.last_uncached if self.cache
                                    else self.prompt_tokens(self.i))
            self.phase = "generating"
        self.current += value.numel()
        self.report("generating speech")

    def end(self):
        pass

    def chunk_started(self):
        self._prompt_seen = False
        self.current = 0
        self.t_chunk = time.time()
        self.phase = "prefill"
        cold = self.i == 0 and not self.warm
        self.report("reading voice + text prompt" if cold else "reading text", force=True)

    def decode_started(self):
        self.phase = "decoding"
        self.report("decoding audio", force=True)

    def chunk_finished(self):
        if self.t_first_token:
            self.gen_time += time.time() - self.t_first_token
        self.finished_tokens.append(self.current)
        self.current = 0
        self.t_first_token = None
        self.i += 1

    def learn(self):
        """Fold this run's measurements into the defaults for the next estimate."""
        tokens = sum(self.finished_tokens)
        if tokens > 20 and self.gen_time > 0:
            _learned["sec_per_token"] = self.gen_time / tokens
        if self.prefill_tokens > 50:
            _learned["prefill_per_token"] = self.prefill_time / self.prefill_tokens
        chars = sum(self.chars)
        if chars and len(self.finished_tokens) == self.n:
            seen = max(0.5, (tokens - TOKENS_FLOOR * self.n) / chars)
            _learned["tokens_per_char"] = 0.5 * _learned["tokens_per_char"] + 0.5 * seen


@contextmanager
def track_progress(m, progress, chunks, reference, ref_text):
    """Temporarily hook a KittenTTS 2 model so each stage reports to `progress`."""
    tracker = ProgressTracker(progress, chunks, getattr(m, "prefix_cache", None))
    hooked = [(m, "_speaker"), (m.model, "generate"), (m.codec, "decode")]
    saved = {(id(obj), name): vars(obj).get(name) for obj, name in hooked}
    orig_speaker, orig_generate, orig_decode = m._speaker, m.model.generate, m.codec.decode

    def speaker(*args, **kwargs):
        if reference and not ref_text:
            # transcription time (and a first-use download) can't be estimated
            progress(None, desc="Transcribing reference clip with Whisper "
                                "(first use downloads Whisper large-v3, ~3 GB)…")
        else:
            tracker.report("analyzing reference clip" if reference else "preparing voice",
                           force=True)
        return orig_speaker(*args, **kwargs)

    def generate(*args, **kwargs):
        tracker.chunk_started()
        return orig_generate(*args, streamer=tracker, **kwargs)

    def decode(*args, **kwargs):
        tracker.decode_started()
        try:
            return orig_decode(*args, **kwargs)
        finally:
            tracker.chunk_finished()

    m._speaker, m.model.generate, m.codec.decode = speaker, generate, decode
    try:
        yield tracker
    finally:
        # put back whatever was there before (the prefix cache's wrapper, or the class method)
        for obj, name in hooked:
            previous = saved[(id(obj), name)]
            if previous is None:
                vars(obj).pop(name, None)
            else:
                setattr(obj, name, previous)
    tracker.learn()


# ---------------------------------------------------------------- KittenTTS 2

def synthesize(text, speaker_mode, voice, ref_audio, ref_text, ref_prompt,
               preset, normalize, stream, override, temperature, top_k, top_p, min_p,
               max_new_tokens, seed, rep_penalty, rep_window, run_penalty, run_grace,
               chunk_chars, chunk_min_chars, chunk_gap, use_emotion, weights, decoder,
               engine, onnx_model, speed, progress=gr.Progress()):
    if not text.strip():
        raise gr.Error("Enter some text first.")
    if engine == ONNX_ENGINE:
        yield synthesize_onnx(text, onnx_model, voice, speed, progress)
        return

    kwargs = dict(preset=preset, normalize=normalize, max_new_tokens=int(max_new_tokens))
    if speaker_mode == "Clone":
        if not ref_audio:
            raise gr.Error("Upload or record a reference clip (5-30 s of one speaker).")
        kwargs["reference"] = ref_audio
        if ref_text.strip():
            kwargs["reference_text"] = ref_text.strip()
        if ref_prompt != "Model default":
            kwargs["use_reference_prompt"] = ref_prompt == "On"
        tag = "clone"
    else:
        if voice not in ALL_VOICES:
            raise gr.Error("Pick a voice first.")
        kwargs["voice"] = voice
        tag = voice

    if override:
        kwargs.update(temperature=temperature, top_k=int(top_k), top_p=top_p, min_p=min_p)

    advanced = dict(repetition_penalty=rep_penalty, token_run_penalty=run_penalty,
                    token_run_grace=int(run_grace), chunk_gap_s=chunk_gap)
    if seed is not None and seed >= 0:
        advanced["seed"] = int(seed)
    for name, value in (("repetition_window", rep_window), ("chunk_chars", chunk_chars),
                        ("chunk_min_chars", chunk_min_chars)):
        if value:
            advanced[name] = int(value)
    if use_emotion != "Auto (on when tags are used)":
        advanced["use_emotion"] = use_emotion == "Force on"
    kwargs["advanced"] = advanced

    key = ("KittenML/kitten-tts-2", (("decoder", decoder), ("weights", weights)))
    if key not in _models:
        progress(None, desc="Loading KittenTTS 2 (first use downloads ~1 GB)…")
    m = get_model("KittenML/kitten-tts-2", weights=weights, decoder=decoder)
    sr = m.sample_rate

    chunks = split_chunks(m, text, normalize, advanced)
    use_ref = kwargs.get("use_reference_prompt", m.use_reference_prompt)
    start = time.time()
    with track_progress(m, progress, chunks,
                        kwargs.get("reference"), kwargs.get("reference_text")) as tracker:
        # resolve the voice first (cached by the model) so the estimate knows
        # whether its prompt still has to be read
        reference = m._speaker(kwargs.get("voice"), kwargs.get("reference"),
                               kwargs.get("reference_text"))
        warm = m.prefix_cache.is_warm(reference, use_ref)
        tracker.set_voice(len(reference["prefix"]) if use_ref else 0, warm)
        note = " · voice already cached" if warm and use_ref else ""
        yield None, (f"Estimated time: ~{fmt_secs(tracker.remaining())} "
                     f"({len(chunks)} chunk{'s' * (len(chunks) > 1)}{note})")
        if stream:
            chunks = []
            for chunk in m.generate_stream(text, **kwargs):
                chunks.append(chunk)
                total = sum(len(c) for c in chunks) / sr
                yield (sr, np.concatenate(chunks)), \
                    f"Chunk {len(chunks)} ready — {total:.1f}s of audio so far…"
            audio = np.concatenate(chunks)
        else:
            audio = m.generate(text, **kwargs)

    path = save(audio, sr, tag)
    took = time.time() - start
    yield (sr, audio), f"Done: {len(audio) / sr:.1f}s of audio in {took:.1f}s. Saved to {path}"


def add_emotion(text, emotion):
    body = text
    if body.startswith("["):
        end = body.find("]")
        if end != -1 and body[1:end] in EMOTIONS:
            body = body[end + 1:].lstrip()
    return body if emotion == "none" else f"[{emotion}] {body}"


def add_event(text, event):
    return f"{text.rstrip()} <{event}> ".lstrip()


# ---------------------------------------------------------------- ONNX models

def synthesize_onnx(text, model_label, voice, speed, progress=gr.Progress()):
    if not text.strip():
        raise gr.Error("Enter some text first.")
    progress(None, desc=f"Loading {model_label}…")
    m = get_model(ONNX_MODELS[model_label])
    progress(None, desc="Generating…")
    start = time.time()
    audio = m.generate(text, voice=voice, speed=speed)
    path = save(audio, 24000, f"onnx-{voice}")
    return (24000, audio), f"Done in {time.time() - start:.1f}s. Saved to {path}"


# ---------------------------------------------------------------- UI

LANGUAGES = {"English": VOICES["English"], "Other": VOICES["Other languages"]}

EXPRESSION_HELP = """
**Emotion.** Put one tag at the very start to set the mood of the whole line, like
`[joyful] We won!`. Pick it from the chips above. Emotion support is early beta: it
steers the delivery but doesn't guarantee it.

**Vocal events.** These can go anywhere in the text, like `We won <laugh> I can't believe it.`
Click a chip to add one at the end.

**Emphasis.** Wrap a word in triple parentheses, like `I told you (((never))) to open that door.`
Select a word and click **Emphasize** to wrap it.
"""

CLONE_HELP = """
Use 5–30 s of **one** speaker with clean audio and no music. 10–20 s works best.
Typing the exact transcript skips Whisper and helps quality a little.
"""

SETTINGS_HELP = """
**Livelier output:** pick *expressive*. To push further, override sampling with
`top_k = 0`, `temperature` around 1.1 and `top_p` of 0.95 or higher.

**Speed:** on CPU or Apple Silicon expect roughly 2–3× slower than real time.
Every result is saved in `outputs/`.
"""

CSS = """
.gradio-container { max-width: 1480px !important; margin: 0 auto !important; }
footer { display: none !important; }

/* header: settings icon left, title centred */
#header { align-items: center; gap: 0; margin: 4px 0 18px; }
#header > * { flex-grow: 0; }
#header #title { flex-grow: 1; }
#settings-btn, #close-settings {
    min-width: 44px !important; max-width: 44px; height: 44px; padding: 0 !important;
    border-radius: 12px !important; font-size: 20px; line-height: 1;
    background: var(--background-fill-primary) !important;
    border: 1px solid var(--border-color-primary) !important; box-shadow: none !important;
}
#settings-btn:hover, #close-settings:hover { background: var(--background-fill-secondary) !important; }
#header-spacer { min-width: 44px !important; max-width: 44px; }
#title { text-align: center; }
#title h1 { margin: 0; font-size: 1.9rem; font-weight: 700; letter-spacing: -0.03em; }
#title p { margin: 2px 0 0; color: var(--body-text-color-subdued); font-size: .92rem; }

/* columns as cards */
.card {
    border-radius: 20px !important; padding: 20px !important; gap: 14px !important;
    background: var(--background-fill-primary) !important;
    border: 1px solid var(--border-color-primary) !important;
    box-shadow: 0 1px 2px rgba(0,0,0,.04), 0 12px 32px -12px rgba(0,0,0,.08);
}
.eyebrow { font-size: .72rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase;
           color: var(--body-text-color-subdued); margin: 0; }
.subtle, .subtle p { color: var(--body-text-color-subdued); font-size: .88rem; }

/* blocks inside cards and the drawer: no grey boxes, plain labels */
:is(.card, .sidebar) .block:not(.info-box) { background: transparent !important; border: none !important;
                              box-shadow: none !important; }
:is(.card, .sidebar) .form, :is(.card, .sidebar) fieldset { background: transparent !important; border: none !important; }
:is(.card, .sidebar) [data-testid="block-info"], :is(.card, .sidebar) label[data-testid="block-label"] {
    background: none !important; border: none !important; padding: 0 !important; margin-bottom: 8px;
    color: var(--body-text-color-subdued) !important; font-size: .72rem; font-weight: 700;
    letter-spacing: .09em; text-transform: uppercase; box-shadow: none !important; }
:is(.card, .sidebar) label[data-testid="block-label"] svg { display: none; }
#tts-text textarea, :is(.card, .sidebar) [data-testid="textbox"] { border: 1px solid var(--border-color-primary) !important;
    border-radius: 14px !important; background: var(--background-fill-primary) !important; }
:is(.card, .sidebar) [data-testid="audio"], :is(.card, .sidebar) .audio-container { border-radius: 14px; }

/* segmented switches (Voice/Clone, English/Other, Engine) */
.segmented { background: none !important; border: none !important; padding: 0 !important; }
.segmented .wrap:has(> label) { display: flex !important; flex-wrap: nowrap !important; gap: 4px !important;
                   padding: 4px; border-radius: 12px; background: var(--background-fill-secondary); }
.segmented label { flex: 1 1 0; justify-content: center; margin: 0 !important; padding: 8px 10px !important;
                   border: none !important; border-radius: 9px !important; background: transparent !important;
                   box-shadow: none !important; cursor: pointer; transition: background .15s, box-shadow .15s; }
.segmented label input { display: none !important; }
.segmented label:has(input:checked) { background: var(--background-fill-primary) !important; font-weight: 600;
                                      color: var(--body-text-color) !important;
                                      box-shadow: 0 1px 3px rgba(0,0,0,.14) !important; }

/* chip pickers (voices, emotions) */
.chips { background: none !important; border: none !important; padding: 0 !important; }
.chips .wrap:has(> label) { gap: 6px !important; }
.chips label { margin: 0 !important; padding: 5px 12px !important; border-radius: 999px !important;
               cursor: pointer; transition: background .15s, color .15s, border-color .15s; }
.chips label input { display: none !important; }
.chips label:has(input:checked) { background: var(--color-accent) !important; color: white !important;
                                  border-color: var(--color-accent) !important; }
.voice-grid .wrap:has(> label) { display: grid !important; grid-template-columns: repeat(auto-fill, minmax(92px, 1fr)); }
.voice-grid, .voice-grid .wrap { overflow: visible !important; max-height: none !important; }
.voice-grid label { justify-content: center; border-radius: 10px !important; padding: 9px 6px !important; }

.event-row { flex-wrap: wrap !important; gap: 6px !important; }
.event-row button { flex: 0 0 auto !important; min-width: 0 !important; border-radius: 999px !important;
                    padding: 4px 11px !important; font-family: var(--font-mono); font-size: .8rem; }

.info-box { border-radius: 14px; padding: 4px 16px !important; background: var(--background-fill-secondary) !important;
            border: none !important; }
.info-box p { font-size: .9rem; margin: 10px 0; }

#tts-text textarea { font-size: 1.02rem; line-height: 1.55; }
#generate-btn { height: 56px; font-size: 1.05rem; font-weight: 650; border-radius: 14px !important; }
#status { min-height: 1.5em; }

/* settings drawer: hide the default chevron, the gear opens it */
.sidebar .toggle-button { display: none !important; }
#settings-head { align-items: center; }
#settings-head h2 { margin: 0; font-size: 1.25rem; }
"""

TOGGLE_SETTINGS = "() => { document.querySelector('.sidebar .toggle-button')?.click(); }"

# wraps the selected text in (((...))); with nothing selected it adds a placeholder
EMPHASIZE = """(t) => {
    const ta = document.querySelector('#tts-text textarea');
    const s = ta ? ta.selectionStart : t.length, e = ta ? ta.selectionEnd : t.length;
    if (s === e) return t.trimEnd() + ' (((word)))';
    return t.slice(0, s) + '(((' + t.slice(s, e) + ')))' + t.slice(e);
}"""


@functools.cache
def voice_clips():
    """Voice name -> the recording each built-in voice was made from (ships with the model)."""
    index = Path(hf_hub_download("KittenML/kitten-tts-2", "voices/voices.json"))
    return {name: index.parent / spec["reference"]
            for name, spec in json.loads(index.read_text()).items()}


def voice_preview(voice):
    """Play the voice's reference clip: instant, no model load or generation needed."""
    clip = voice_clips().get(voice)
    if clip is None:
        return None
    audio, sr = sf.read(clip, dtype="float32")
    return sr, audio


def voice_list(engine, language, current):
    choices = ONNX_VOICES if engine == ONNX_ENGINE else LANGUAGES[language]
    value = current if current in choices else ("Bruno" if "Bruno" in choices else choices[0])
    return gr.update(choices=choices, value=value)


def on_language(engine, language, current):
    return voice_list(engine, language, current), gr.update(value=language == "English")


def on_engine(engine, language, mode, current):
    onnx = engine == ONNX_ENGINE
    language = "English" if onnx else language
    return (gr.update(value=language, interactive=not onnx),
            gr.update(value="Voice" if onnx else mode, interactive=not onnx),
            voice_list(engine, language, current),
            gr.update(visible=not onnx),     # expression tools
            gr.update(visible=not onnx),     # KittenTTS 2 settings
            gr.update(visible=onnx))         # ONNX settings


def on_mode(mode):
    return gr.update(visible=mode == "Voice"), gr.update(visible=mode == "Clone")


with gr.Blocks(title="KittenTTS Studio") as demo:

    # ------------------------------------------------------------ settings drawer
    with gr.Sidebar(open=False, width=400, elem_id="settings"):
        with gr.Row(elem_id="settings-head"):
            gr.Markdown("## Settings")
            close_settings = gr.Button("✕", elem_id="close-settings", scale=0)

        engine = gr.Radio([K2_ENGINE, ONNX_ENGINE], value=K2_ENGINE, label="Engine",
                          info="Lightweight: the small 0.8 models. Fast on CPU, "
                               "8 English voices, speed control, no cloning or tags.",
                          elem_classes="segmented")

        with gr.Group(visible=False) as onnx_settings:
            onnx_model = gr.Dropdown(list(ONNX_MODELS), value="mini (80M, best)", label="Model")
            speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Speed")

        with gr.Column(visible=True) as k2_settings:
            preset = gr.Radio(list(PRESETS), value="stable", label="Preset",
                              elem_classes="segmented")
            normalize = gr.Checkbox(value=True, label="Normalize text",
                                    info="Spells out numbers etc. Turned off automatically "
                                         "when you switch to Other.")
            stream = gr.Checkbox(value=False, label="Stream",
                                 info="Play each chunk as soon as it's ready")

            with gr.Accordion("Sampling", open=False):
                override = gr.Checkbox(value=False, label="Override preset values")
                temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="temperature")
                top_k = gr.Slider(0, 200, value=50, step=1, label="top_k (0 = off)")
                top_p = gr.Slider(0.1, 1.0, value=0.8, step=0.01, label="top_p")
                min_p = gr.Slider(0.0, 0.5, value=0.0, step=0.01, label="min_p")
                seed = gr.Number(value=-1, precision=0,
                                 label="Seed (-1 = random; same seed = same audio)")
                max_new_tokens = gr.Slider(100, 3000, value=1000, step=50,
                                           label="Max tokens per chunk (25 ≈ 1 s)")

            with gr.Accordion("Voice cloning", open=False):
                ref_prompt = gr.Radio(["Model default", "On", "Off"], value="Model default",
                                      label="Include clip audio in prompt",
                                      elem_classes="segmented")

            with gr.Accordion("Advanced", open=False):
                rep_penalty = gr.Slider(1.0, 2.0, value=1.1, step=0.05, label="repetition_penalty")
                rep_window = gr.Number(value=0, precision=0, label="repetition_window (0 = default)")
                run_penalty = gr.Slider(1.0, 3.0, value=1.3, step=0.05, label="token_run_penalty")
                run_grace = gr.Slider(1, 50, value=10, step=1, label="token_run_grace")
                chunk_chars = gr.Number(value=0, precision=0, label="chunk_chars (0 = default)")
                chunk_min_chars = gr.Number(value=0, precision=0,
                                            label="chunk_min_chars (0 = default)")
                chunk_gap = gr.Slider(0.0, 1.0, value=0.16, step=0.01,
                                      label="Pause between chunks (s)")
                use_emotion = gr.Radio(["Auto (on when tags are used)", "Force on", "Force off"],
                                       value="Auto (on when tags are used)",
                                       label="Emotion conditioning")

            with gr.Accordion("Model", open=False):
                weights = gr.Radio(["packed", "emb4", "full"], value="packed", label="Weights",
                                   info="packed = lossless 947 MB · emb4 = 506 MB, slightly "
                                        "lower quality · full = bf16")
                decoder = gr.Radio(["default", "student_w4"], value="default", label="Decoder",
                                   info="student_w4 is 39 MB vs 459 MB, lower quality, not faster")

            gr.Markdown(SETTINGS_HELP, elem_classes="subtle")

    # ------------------------------------------------------------ header
    with gr.Row(elem_id="header", equal_height=True):
        settings_btn = gr.Button("⚙", elem_id="settings-btn", scale=0)
        gr.HTML("<h1>🐱 KittenTTS Studio</h1><p>Local text-to-speech with voice cloning</p>",
                elem_id="title")
        gr.HTML("", elem_id="header-spacer")

    with gr.Row(equal_height=False):

        # -------------------------------------------------------- left: voice
        with gr.Column(scale=4, min_width=300, elem_classes="card"):
            gr.HTML("<p class='eyebrow'>Voice</p>")
            speaker_mode = gr.Radio(["Voice", "Clone"], value="Voice", show_label=False,
                                    container=False, elem_classes="segmented")
            language = gr.Radio(["English", "Other"], value="English", label="Language",
                                elem_classes="segmented")

            with gr.Column(visible=True) as builtin_box:
                voice = gr.Radio(LANGUAGES["English"], value="Bruno", show_label=False,
                                 container=False, elem_classes=["chips", "voice-grid"])
                preview = gr.Audio(label="Preview", type="numpy", interactive=False,
                                   autoplay=True, elem_id="preview")

            with gr.Column(visible=False) as clone_box:
                ref_audio = gr.Audio(sources=["upload", "microphone"], type="filepath",
                                     label="Reference clip (5–30 s, one speaker)")
                ref_text = gr.Textbox(label="Transcript of the clip (optional)", lines=3,
                                      placeholder="Leave empty to auto-transcribe with Whisper")
                gr.Markdown(CLONE_HELP, elem_classes="subtle")

        # -------------------------------------------------------- middle: text
        with gr.Column(scale=6, min_width=340, elem_classes="card"):
            gr.HTML("<p class='eyebrow'>Text</p>")
            text = gr.Textbox(
                show_label=False, lines=9, max_lines=30, elem_id="tts-text",
                placeholder="Type what you want to hear…",
                value="[joyful] One day, a little girl named Lily found a needle in her room "
                      "<gasp> and she was (((so))) curious about it.")

            with gr.Column(visible=True) as expression_box:
                emotion = gr.Radio(["none"] + EMOTIONS, value="none", label="Emotion",
                                   elem_classes="chips")
                gr.HTML("<p class='eyebrow' style='margin-top:4px'>Vocal events</p>")
                with gr.Row(elem_classes="event-row"):
                    for e in EVENTS:
                        gr.Button(f"<{e}>", size="sm").click(
                            None, text, text, js=f"(t) => (t.trimEnd() + ' <{e}> ').trimStart()")
                    emphasize = gr.Button("((( Emphasize )))", size="sm", variant="secondary")
                emphasize.click(None, text, text, js=EMPHASIZE)
                gr.Markdown(EXPRESSION_HELP, elem_classes="info-box")

        # -------------------------------------------------------- right: output
        with gr.Column(scale=4, min_width=300, elem_classes="card"):
            gr.HTML("<p class='eyebrow'>Output</p>")
            go = gr.Button("Generate", variant="primary", elem_id="generate-btn")
            out_audio = gr.Audio(label="Result", type="numpy", interactive=False)
            status = gr.Markdown("Pick a voice, write some text and hit **Generate**.",
                                 elem_id="status", elem_classes="subtle")

    # ------------------------------------------------------------ events
    settings_btn.click(None, js=TOGGLE_SETTINGS)
    close_settings.click(None, js=TOGGLE_SETTINGS)

    speaker_mode.change(on_mode, speaker_mode, [builtin_box, clone_box])
    voice.input(voice_preview, voice, preview)
    language.change(on_language, [engine, language, voice], [voice, normalize])
    engine.change(on_engine, [engine, language, speaker_mode, voice],
                  [language, speaker_mode, voice, expression_box, k2_settings, onnx_settings])
    emotion.input(add_emotion, [text, emotion], text)
    preset.change(lambda p: [gr.update(value=v) for v in PRESETS[p].values()],
                  preset, [temperature, top_p, top_k, min_p])

    go.click(synthesize,
             [text, speaker_mode, voice, ref_audio, ref_text, ref_prompt,
              preset, normalize, stream, override, temperature, top_k, top_p, min_p,
              max_new_tokens, seed, rep_penalty, rep_window, run_penalty, run_grace,
              chunk_chars, chunk_min_chars, chunk_gap, use_emotion, weights, decoder,
              engine, onnx_model, speed],
             [out_audio, status])

if __name__ == "__main__":
    theme = gr.themes.Soft(primary_hue="violet", secondary_hue="slate", neutral_hue="slate",
                           radius_size="lg", font=[gr.themes.GoogleFont("Inter"), "system-ui"])
    demo.queue().launch(inbrowser=True, theme=theme, css=CSS)
