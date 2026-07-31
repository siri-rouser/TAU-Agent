# RAG_Retrieve
This folder contains the agentic retrieval pipeline for TAU-Agent.

## Prior Prep

Download the following checkpoints and place them under `RAG_retriever/weights/`:

- `yolo26x.pt` — generic COCO-class YOLO, used as the default object detector.
- `last.pt` — the vehicle color/type/subtype classifier checkpoint (dinov2-based;
  see [vehicle_classifier](../vehicle_classifier/README.md)). Despite the
  filename, this is **not** a YOLO checkpoint.

## `caption_generation.py` — generate captions locally (used to produce `data/captions`)

Generates per-video captions (segments, summaries, and scene descriptions) used as RAG evidence. The script calls the Gemini API through the [Modellsell Platform](https://modelsell.com/) and requires `MODELSELL_API_KEY` to be set.

Example Run:
```bash
python RAG_retriever/caption_generation.py \
    --data-json data/dataset/train/track3/bcq.json \
    --media-root data/videos --base-dir data/captions
```

## `main.py` — run the RAG retriever on a single video

Example script showing how the agentic retrieval pipeline works end-to-end:
it captions the video (if not already cached), then lets the main agent
call the `caption_retrieval` and `free_text_tracking` tools to answer your
question(s). Requires `MODELSELL_API_KEY` set.

Example Run:
```bash
python RAG_retriever/main.py \
    --video-path data/videos/Accident-Bench/land_space/medium/videos/MxiC2UdOzE8.mp4 \
    --question "Does the gray sedan drive straight through the intersection without turning?\nAnswer with Yes or No followed by a brief explanation." \
    --output-dir RAG_retriever/results
```

## `RAG_stage1.py` — run the RAG retriever for all question-answer pairs 

Create offline agentic RAG results for both train and test set of data. It mainly output "relevant_frame_ranges", "segment_captions" and "relevant_tracks_data" from question-query.

Example Run:
```bash
python RAG_retriever/RAG_stage1.py \
    --split test \
    --tasks test \
    --captioning-agent Gemini31pro \
    --dataset-dir data/dataset/test/tar_test \
    --output-dir data/RAG_Info/test/tar_test \
    --gpus 0 \
    --workers-per-gpu 4

python RAG_retriever/RAG_stage1.py \
    --split train \
    --tasks bcq \
    --captioning-agent Gemini35Flash \
    --dataset-dir data/dataset/train/track3 \
    --output-dir data/RAG_Info/train \
    --gpus 0 \
    --workers-per-gpu 4
```

## `RAG_stage2.py` — enrich stage-1 RAG JSONs with video-level cross-question context

Run optional cross-question context retrieve to get context information from other questions, it will output "factual_information" and "potential_information" frm cross-question context.

Example Run:
```bash
python RAG_retriever/RAG_stage2.py --split train
python RAG_retriever/RAG_stage2.py --split test
```