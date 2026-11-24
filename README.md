# LangGraspRefine

This project predicts a 2D grasp from an RGB image and a text instruction, such as “grasp the mug by its handle.” A FiLM heatmap CNN proposes 24 grasp boxes. A pretrained Grounding DINO model finds the object and, where possible, the named part. The baseline uses those detections to rank the original boxes.

I added a small MLP that looks at the pixels around each box, the detector scores, and the other proposals for that instruction. It learns to move, resize, and rotate each box and to score the corrected box. Both rows in the results table start with the same 24 proposals and use the same frozen detector.

## Where to find the code

| Path | What it does |
|---|---|
| [geometry_refinement/model.py](geometry_refinement/model.py) | Heatmap CNN, training loss, and decoding of the 24 starting boxes |
| [geometry_refinement/refinement.py](geometry_refinement/refinement.py) | Applies the five predicted changes to each grasp box |
| [common/prepare_full_split.py](common/prepare_full_split.py) | Builds the scene splits using the dataset ZIPs and the category metadata in `12_full_scale/assets/` |
| [common/train_full.py](common/train_full.py) | Trains the heatmap CNN and saves its checkpoints |
| [common/grounding.py](common/grounding.py) | Finds the object and searches a crop for the named part |
| [common/candidate_features.py](common/candidate_features.py) | Samples pixels around each box and matches candidates to training labels |
| [common/learned.py](common/learned.py) | Defines and trains the refinement MLP |
| [12_full_scale/panels.py](12_full_scale/panels.py) | Makes data shards and caches object and part detections |
| [12_full_scale/run_2d.py](12_full_scale/run_2d.py) | Fits the MLP, evaluates both methods, and gathers the metrics |
| [12_full_scale/pipeline.py](12_full_scale/pipeline.py) | Scores the starting and corrected boxes and sorts them |
| [analysis/](analysis/) | Checks saved results and draws before-and-after examples |
| [reported_results.json](reported_results.json) | Exact counts, validation settings, and split checksums from the paper |

## Results

| Method | Seen Top-1 | Unseen Top-1 | Top-1 H | Top-5 H |
|---|---:|---:|---:|---:|
| Grounded heatmap | 66.32% | 49.18% | 56.48% | 82.57% |
| + learned refinement | 70.17% | 53.44% | 60.67% | 82.90% |

Refinement raised the Top-1 harmonic mean by **4.19 percentage points**. On Seen it matched 146,035 of 208,116 instructions at rank one. On Unseen it matched 65,614 of 122,788.

A box counts as correct if it has both rotated IoU greater than 0.25 and angle error below 30° against the same grasp annotation for that instruction. Top-1 checks the first box. Top-5 checks whether any of the first five boxes is correct. H is the harmonic mean of the Seen and Unseen rates. These scores measure agreement with the 2D annotations. [Full counts and settings](reported_results.json)

## Set up the data

Use Python 3.10 or newer. For full training and detection, use a CUDA-enabled PyTorch build that matches your GPU. Run the commands below from the repository root. [PyTorch installation](https://pytorch.org/get-started/locally/)

~~~sh
python -m pip install -r requirements.txt
~~~

Extract the original RGB images into `data/images`. Keep the scene descriptions, grasp instructions, and positive labels in their ZIP archives. The links below point to the dataset revisions used by the split script. [Grasp-Anything RGB and descriptions](https://huggingface.co/datasets/airvlab/Grasp-Anything/tree/6b4917455d0910c23f936a6c7a968418ba518897) · [Grasp-Anything++ instructions and labels](https://huggingface.co/datasets/airvlab/Grasp-Anything-pp/tree/9b30c841d4f04e4a62e9f7abb3b503337a6200a7)

Download the frozen detector with the Hugging Face CLI.

~~~sh
hf download IDEA-Research/grounding-dino-tiny \
  --revision a2bb814dd30d776dcf7e30523b00659f4f141c71 --local-dir data/grounder
~~~

Place `train.jsonl`, `val.jsonl`, `test_seen.jsonl`, and `test_unseen.jsonl` in `data/splits`, or make a new split with the commands below. Each line is one instruction and has `scene_id`, `target_id`, `image`, `prompt`, and a nonempty list of `grasps`. A grasp is `[cx, cy, width, height, angle]` in original image coordinates, with the angle in radians. Set `--scene-descriptions` to the local path of the scene-description ZIP.

~~~sh
python common/prepare_full_split.py inventory --work work/inventory \
  --images data/images --scene-descriptions data/scene_descriptions.zip \
  --instructions-zip data/grasp_instructions.zip --positive-zip data/grasp_label_positive.zip \
  --taxonomy 12_full_scale/assets/lvis_v1_categories.json
python common/prepare_full_split.py crosscheck --work work/inventory
python common/prepare_full_split.py rebalance --work work/inventory --seed 20261121
python common/prepare_full_split.py plan --work work/inventory --seed 20261121
python common/prepare_full_split.py materialize --work work/inventory --out data/splits --workers 4
~~~

The paper used 2,152,445 training instructions from 483,581 scenes. Validation, Seen test, and Unseen test each had 40,000 separate scenes. The Unseen test instructions targeted held-out classes. To match the paper split exactly, use the original split files and verify their SHA-256 checksums. To rebuild it, also supply the same earlier-scene exclusions to both `rebalance` and `plan` using `--history-dir`, `--prior-used`, or `--protect`. A fresh split with the same seed and different exclusions can produce different results. [Paper split checksums](reported_results.json)

## Train and evaluate

| Setting | Heatmap CNN | Refinement MLP |
|---|---|---|
| Input | RGB at `160 × 160`, up to 20 text tokens | 161 values per candidate, including a rotated `7 × 7` RGB grid |
| Batch size | 64 instructions | 96 groups of 24 proposals |
| Epochs | 12 | Up to 14 |
| Optimizer | AdamW, learning rate `0.0004`, weight decay `0.02` | AdamW, learning rate `0.002`, weight decay `0.01` |
| Learning rate schedule | Cosine to `0.000032` | Cosine to `0.0002` |
| Random seed | 121 | Fit seed 121 gives a refiner seed of 122 |
| Checkpoint used in the paper | Epoch 9, highest validation Top-1 | Epoch 14, lowest validation loss |

The refiner uses the training annotations to learn box corrections and scores while the CNN and detector stay frozen. Its loss adds binary cross-entropy, 0.6 times listwise loss, and 3 times the box loss. The fit stage uses 8,192 validation instructions to pick the refiner checkpoint, calibrate its scores, and set the scoring weights. The test groups are scored after those choices are fixed.

In the paper run, the baseline adds 8 times the object score and 0.5 times the cropped-part score to the heatmap rank term. The refiner applies the full box correction, recomputes those two scores, then adds 1.6 times its calibrated score. [Saved settings](reported_results.json)

~~~sh
python common/train_full.py --method geometry_refinement/model.py --images data/images \
  --splits data/splits --output geometry_refinement/full_v1 \
  --epochs 12 --batch 64 --workers 8 --seed 121 --lr 0.0004 --skip-test

python 12_full_scale/panels.py prepare --splits data/splits --output work/panels \
  --all-train --val-count 8192 --shard-size 2048
python 12_full_scale/panels.py cache --panels work/panels --images data/images \
  --model data/grounder --output work/cache
python 12_full_scale/run_2d.py fit --root . --tag full_v1 --panels work/panels \
  --cache work/cache --images data/images --output work/fit --seed 121
python 12_full_scale/run_2d.py evaluate --root . --tag full_v1 --fit work/fit \
  --panels work/panels --cache work/cache --images data/images --output work/evaluation
python 12_full_scale/run_2d.py summarize --panels work/panels --fit work/fit \
  --evaluation work/evaluation --output work/metrics.json
~~~

- `geometry_refinement/full_v1/best.pt` holds the selected CNN. Use the same CNN training command with `--resume` to continue from `last.pt`, keeping the original training inputs and config.
- `work/fit/pose_refiner.pt` holds the trained MLP. The same directory contains `validation_selection.json` for the chosen weights and `learner_history.json` for the loss history.
- `work/evaluation` holds per-instruction predictions. `work/metrics.json` summarizes both methods.
- To share detection caching or evaluation across jobs, set `--workers N` and use one `--worker` index from 0 through `N-1` per job. Keep the same input files when restarting. The full fit loads the proposal records into memory. Use `--train-count` in place of `--all-train` for a smaller training check.

## Check the saved results

Run the audit to verify the evaluation shards and count how many first predictions went from wrong to right or right to wrong. To draw examples, replay selected predictions with the saved models and render the boxes before and after refinement.

~~~sh
python analysis/collect_paired_counts.py --panels work/panels --evaluation work/evaluation
python analysis/collect_visual_examples.py --root . --tag full_v1 --panels work/panels \
  --cache work/cache --fit work/fit --evaluation work/evaluation --images data/images \
  > work/traced_examples.json
python -m pip install "matplotlib>=3.7"
python analysis/render_visual_examples.py --trace work/traced_examples.json \
  --images data/images --output work/qualitative
~~~

To replay the original run, pass its root as `--root`. Keep the matching heatmap source, vocabulary, CNN checkpoint, and refiner checkpoint so the saved file checks can pass.

The earlier experiment code is saved in a separate private repository. [Experiment archive](https://github.com/SILVIO-ZHENG/lang-grasp-experiments-archive)

## References

- [Language-driven Grasp Detection](https://openaccess.thecvf.com/content/CVPR2024/html/Vuong_Language-driven_Grasp_Detection_CVPR_2024_paper.html)
- [FiLM](https://arxiv.org/abs/1709.07871)
- [Grounding DINO](https://arxiv.org/abs/2303.05499)
- [Grasp detection and segmentation with a refinement head](https://doi.org/10.1109/ICRA48506.2021.9561398)
- [LVIS category metadata](12_full_scale/assets/lvis_v1_source.json)
