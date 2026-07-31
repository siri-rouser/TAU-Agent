#!/bin/bash

# Step0: Generate Predictions by Question-Answering VLM
python eval/eval_aicity_rag_test.py \
--model-dir /output/model_checkpoint --lora \
--rag-dir data/RAG_Info/test/tar_test/tar_test \
--test-json data/dataset/test/tar_test/test.json \
--video-dir data/videos \
--output-dir eval/aicity_test \
--gemini-caption-dir data/captions/gemini31/tar_test \
--shard-rank 0 --shard-size 1 &

# Step1: Create vote for mcq,bcq,mcq_openended,bcq_openended
python eval/vote_rag_mcq_bcq.py \
    --model-dir /output/model_checkpoint --lora \
    --test-json data/dataset/test/tar_test/test.json \
    --rag-dir data/RAG_Info/test/tar_test/test \
    --video-dir data/videos \
    --tasks mcq,bcq \
    -o eval/output/vote_mcqbcq

python eval/vote_rag_mcq_bcq.py \
    --model-dir /output/model_checkpoint --lora \
    --test-json data/dataset/test/tar_test/test.json \
    --rag-dir data/RAG_Info/test/tar_test/test \
    --video-dir data/videos \
    --tasks mcq_openended,bcq_openended \
    -o eval/output/vote_mcqbcq_oe

# Step2: Post process the vote result to get final result
python eval/fix_lowconf_rag_bcq.py \
    --model-dir /output/model_checkpoint --lora \
    --predictions eval/output/vote_mcqbcq/predictions_voted.jsonl,eval/output/vote_mcqbcq_oe/predictions_voted.jsonl \
    --descriptions eval/aicity_test/predictions/rag_test_predictions.jsonl \
    --rag-dir data/RAG_Info/test/tar_test/test \
    --video-dir data/videos --tasks bcq,bcq_openended \
    --force-bcq \
    --bcq-ctx-fields temporal,causal,summary,open_qa \
    -o eval/output/bcq_fixed

python eval/fix_lowconf_rag_mcq.py \
    --model-dir /output/model_checkpoint --lora \
    --predictions eval/output/bcq_fixed/predictions_lowconf_fixed.jsonl \
    --descriptions eval/aicity_test/predictions/rag_test_predictions.jsonl \
    --rag-dir data/RAG_Info/test/tar_test/test \
    --probe-fault \
    --video-dir data/videos \
    --gpus 0,1,2 \
    --ctx-fields scene \
    -o eval/output/mcq_fixed

# Step3: MCQ, MCQ_OPENEND alignment
python eval/harmonize_oe_to_mcq.py \
    --predictions eval/output/mcq_fixed/predictions_mcq_fixed.jsonl \
    -o eval/output/mcq_fixed/predictions_oe_harmonized.jsonl

# Step4: Re-Rank 
python eval/regen_freetext_consensus.py \
    --model-dir /output/model_checkpoint --lora \
    --predictions eval/aicity_test/predictions/rag_test_predictions.jsonl \
    --rag-dir data/RAG_Info/test/tar_test/test \
    --video-dir data/videos \
    --num-samples 5 --temperature 0.7 \
    -o eval/output/freetext_consensus

# Step5: Make submission Final

python eval/make_submission_final.py \
    --mcq-fixed eval/output/mcq_fixed/predictions_oe_harmonized.jsonl \
    --freetext eval/output/freetext_consensus/predictions_freetext_consensus.jsonl \
    --test-json data/dataset/test/tar_test/test.json \
    -o eval/final_submission.csv