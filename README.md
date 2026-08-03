# OmniMIA

This repository contains the gray-box reference implementation of OmniMIA
from *Detecting Training Data of Omni-Modality Diffusion Language Models*.

1. Randomly mask semantic tokens independently for `T` stochastic trajectories.
2. Record the fixed top-`d` probability vector at every denoising step and
   semantic-token position.
3. Build the trajectory-consistency tensor `F[s, p, j]` with cosine similarity
   over every unordered trajectory pair.
4. Aggregate expected, pessimistic, and optimistic evidence using the paper's
   hierarchical top-`k` aggregation, then add the three views into a membership
   score.
5. Fuse standardized evidence from complementary cross-modal pathways.

OmniMIA supports gray-box trajectory features and a black-box
semantic-stability variant. The black-box implementation is in `blackbox/`.

## Repository layout

```text
omnimia/
  trajectory.py          # Equations 2-4: F[s, p, j] and three-view aggregation
  evaluation.py          # Paper-style cross-modal evidence fusion
  pathways/
    image.py             # Text-to-image and image-to-text probes
    text.py              # Text-to-text probe
    image_edit.py        # Image-to-image probe
    video.py             # Video-to-video continuation probe
    video_aggregate.py   # Video trajectory aggregation
```

## Data

All datasets are stored under `data/`. To keep the repository size manageable,
only a subset of the data is included here.

## Run black-box OmniMIA

The black-box implementation uses only decoded API outputs. For every probing
task, submit the same template-augmented input under independent stochastic
executions, encode the returned content with an external semantic encoder (for
example CLIP for images or BGE-M3 for text), and pass the encoder to
`query_and_extract`. The returned value is the direct membership score; no
target-model logits, tokenizer, hidden states, or trained fusion model are
used.

This final-step example assumes `call_target_api` returns one decoded output
per call and `encode_outputs` returns an array of shape
`[number_of_outputs, embedding_dimension]`:

```python
from blackbox import AccessMode, query_and_extract

sample = {"image": "suspected.png", "caption": "a public caption"}
pathways = ("image_to_text", "text_to_image")

def build_query(sample, pathway):
    # Combine the suspected sample with this pathway's public template.
    return {"pathway": pathway, "sample": sample}

def call_target_api(query, pathway, mode):
    # Make one independent stochastic API request and return decoded content.
    return target_service.generate(query)

encoders = {
    "image_to_text": encode_outputs,
    "text_to_image": encode_outputs,
}

score, names = query_and_extract(
    sample,
    pathways,
    build_query,
    call_target_api,
    encoders,
    repeats=4,
    mode=AccessMode.FINAL_STEP,
    k=2,
)
assert names == ("membership_score",)
print(float(score[0]))
```

For multi-step access, set `mode=AccessMode.MULTI_STEP` and make each API call
return an ordered, equally sized sequence of decoded outputs—one output for
each exposed generation step. The encoder is applied to all returned outputs.
`k` retains the lowest/highest consistency values for the pessimistic and
optimistic views; in multi-step mode it must not exceed either the number of
independent-query pairs or the number of exposed steps. Higher scores indicate
stronger membership evidence.

## Run gray-box pathways

The paper defaults are `mask_ratio=0.5`, `num_trajectories=4`, `steps=18`, and
`aggregation_topk=32`. `top_d` below is the probability-vector truncation
dimension `d` from Section 3.3.

```bash
python -m omnimia.pathways.image \
  --task t2i \
  --input_path samples.jsonl \
  --output_dir outputs/t2i \
  --pretrained_model_path /path/to/omni-model \
  --vq_model_name /path/to/vq-model \
```

```bash
python -m omnimia.pathways.image \
  --task i2t \
  --input_path samples.jsonl \
  --output_dir outputs/i2t \
  --pretrained_model_path /path/to/omni-model \
  --vq_model_name /path/to/vq-model \
```

```bash
python -m omnimia.pathways.text \
  --input_path text_samples.jsonl \
  --output_dir outputs/t2t \
  --pretrained_model_path /path/to/omni-model \
```

```bash
python -m omnimia.pathways.image_edit \
  --input_path edit_samples.jsonl \
  --output_dir outputs/i2i \
  --pretrained_model_path /path/to/omni-model \
  --vq_model_name /path/to/vq-model \
```

Each runner produces `results.jsonl`. A successful record has this stable
schema:

```json
{
  "sample_id": 7,
  "label": 1,
  "membership_score": 2.61,
  "trajectory_evidence": {
    "expected": 0.89,
    "pessimistic": 0.78,
    "optimistic": 0.94
  }
}
```

Intermediate NPZ artifacts use explicit field names:
`trajectory_probabilities`, `semantic_position_mask`, and
`observed_token_log_probabilities`.

Video-to-video uses the same pathway namespace. It requires the sibling
`Omni-Video-main/` checkout and CUDA:

```bash
python -m omnimia.pathways.video \
  --ckpt_dir /path/to/OmniVideo2-1.3B \
  --input_jsonl video_samples.jsonl \
  --output_root outputs/v2v \
  --prefix_frames 10 --continuation_frames 7 --num_repeats 4 --gpu_ids 0

python -m omnimia.pathways.video_aggregate outputs/v2v \
  --output outputs/v2v_fused.json --aggregation-topk 32
```

The video runner stores only `gen_gen` rows: independent generation pairs
required by the trajectory-consistency statistic.

## Fuse cross-modal evidence

Run complementary pathways over the same input order, then fuse their three
views as specified in Section 3.5:

```bash
python -m omnimia.evaluation \
  --pathway-results outputs/t2i/results.jsonl outputs/i2t/results.jsonl \
  --output outputs/t2i_i2t_fused.json
```

The aggregation is intentionally label-free: it sums expected, pessimistic,
and optimistic evidence for every supplied pathway. Labels are used only to
report ROC-AUC.

## Environment

The gray-box image and text pathways depend on the model code exposed as
`models.MMaDA` or `models.LuminaDiMOO`, plus `numpy`, `torch`, `tqdm`, and
`Pillow`. The video pathway additionally needs `decord`, `torchvision`, and
`transformers` with the Omni-Video dependencies.

Install the required dependencies in your environment. The target model
repositories and checkpoints are supplied as external paths rather than being
vendored into this repository.
