"""Quick CLI for KittenTTS.

Usage:
    .venv/bin/python speak.py "Hello there" [-v Bruno] [-o output.wav] [--small]
"""
import argparse

from kittenml import KittenTTS

p = argparse.ArgumentParser()
p.add_argument("text")
p.add_argument("-v", "--voice", default="Bruno")
p.add_argument("-o", "--out", default="output.wav")
p.add_argument("--small", action="store_true", help="use the 506 MiB emb4 weights")
p.add_argument("--list-voices", action="store_true")
p.add_argument("-r", "--ref", help="5-30s WAV of one speaker to clone (overrides --voice)")
p.add_argument("--ref-text", help="transcript of --ref (otherwise Whisper transcribes it)")
args = p.parse_args()

m = KittenTTS("KittenML/kitten-tts-2", **({"weights": "emb4"} if args.small else {}))
if args.list_voices:
    print("\n".join(m.available_voices))
if args.ref:
    speaker = {"reference": args.ref}
    if args.ref_text:
        speaker["reference_text"] = args.ref_text
else:
    speaker = {"voice": args.voice}
m.generate_to_file(args.text, args.out, **speaker)
print(f"Wrote {args.out}")
