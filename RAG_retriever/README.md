Content in this folder are the Agentic Retrieval Pipeline in TAU-Agent.

## `caption_generation.py` — generate all captions locally(We use this to generate /data/captions)

Generates per-video captions (segments + summary + scene description) used
as RAG evidence, via the Gemini API. Requires `MODELSELL_API_KEY` set.

```bash
python RAG_retriever/caption_generation.py \
    --data-json data/dataset/train/track3/bcq.json \
    --media-root /data --base-dir /workspace/TAU-Agent/data/captions
```

Use `--video-dir` instead of `--data-json` for datasets with no annotation
file (e.g. FETV). Already-captioned videos are skipped automatically, so
it's safe to re-run. See `--help` for all options.
