#!/bin/bash
# PSI-VQA (Track 8) inference.
#
# PSI-VQA is evaluated zero-transfer with the same QA-VLM used for TAR. The only
# benchmark-specific choice is that the cross-question context is applied
# task-selectively: kept for open_qa (where it supplies the candidate crossing-
# intent cues) and withheld for bcq/mcq (where it biases the discriminative
# answer). See eval/build_psi_mixed_rag.py for the rationale.

set -e

CKPT=/output/model_checkpoint
PSI_Q=data/dataset/test/PSI_VQA
RAG_BASE=data/RAG_Info/test/PSI_VQA
RAG_CTX=data/RAG_Info/test/PSI_VQA_ctx
RAG_MIXED=data/RAG_Info/test/PSI_VQA_mixed
TEST_JSON=$PSI_Q/test.json
OUT=eval/output_psi

# Step 0: merge the four per-task question files into one combined test JSON
python eval/make_psi_test_json.py --questions-dir $PSI_Q --out $TEST_JSON

# Step 1: rebuild the cross-question context deterministically (no API) from the
#         questions and inject it into the RAG evidence
python RAG_retriever/build_stage2_ctx_noapi.py \
    --questions-dir $PSI_Q --rag-in $RAG_BASE --rag-out $RAG_CTX

# Step 2: keep the context only for open_qa, strip it for bcq/mcq/temporal
python eval/build_psi_mixed_rag.py --ctx-dir $RAG_CTX --out $RAG_MIXED

# Step 3: run the question-answering VLM
python eval/eval_aicity_rag_test.py \
    --model-dir $CKPT --lora \
    --rag-dir $RAG_MIXED \
    --test-json $TEST_JSON \
    --video-dir data/videos \
    --output-dir $OUT \
    --shard-rank 0 --shard-size 1

# Step 4: build the submission CSV (mcq -> single option letter)
python eval/make_psi_submission.py \
    --predictions $OUT/predictions/rag_test_predictions.jsonl \
    --out $OUT/submission_psi.csv
