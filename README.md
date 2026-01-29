# OmniMIA
## OmniMIA dataset
In `./data` path, split by member and non-member

## Run OmniMIA
1. Run T2I task
```
python text2img_image_vocab_similarity_auc.py \
  --input_path <path-to-your-input-file> \
  --output_dir <path-to-your-output-dir> \
  --pretrained_model_path <path-to-your-Omni-model> \
  --vq_model_name <path-to-your-vq-model>\
  --batch_size 1 \
  --mask_ratio 0.5 \
  --topk 32 \
  --steps 18
```
2. Run I2T task
```
python img2text_image_vocab_similarity_auc.py \
  --input_path <path-to-your-input-file> \
  --output_dir <path-to-your-output-dir> \
  --pretrained_model_path <path-to-your-Omni-model> \
  --vq_model_name <path-to-your-vq-model>\
  --batch_size 1 \
  --mask_ratio 0.5 \
  --topk 32 \
  --steps 18
```
3. Aggregate feature of I2T task and T2I task