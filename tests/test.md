python scripts/bench_caption_batch.py \
      --frames-dir data/odc_test_60s/frames \
      --num-frames 16 \
      --batch-sizes 1,2,4,8 \
      --max-tokens 512 \
      --output bench_122b_odc \
      --warmup