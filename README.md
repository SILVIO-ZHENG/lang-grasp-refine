# LangGraspRefine

Given an RGB image and a text instruction such as "grasp the mug by its handle," this project ranks two-dimensional grasp rectangles. It contains two related methods for Grasp-Anything++.

- **Geometric refinement.** A text-conditioned heatmap proposes 24 rectangles. Grounding DINO supplies object and part support. A small MLP uses that support and the pixels near each candidate to adjust its center, size, and angle.
- **Dual-contact proposals with learned ranking.** A second model predicts a pair of gripper contact points, which gives 24 additional candidates after conversion to rectangles. A learned ranker adjusts the refined heatmap scores. The two branches then compete in a shared ranking.

## Reported evaluation

| Method | Seen Top-1 | Unseen Top-1 | Top-1 H | Top-5 H |
|---|---:|---:|---:|---:|
| Grounded heatmap baseline | 66.32 | 49.18 | 56.482 | 82.57 |
| Geometric refinement | 70.17 | 53.44 | 60.671 | 82.90 |
| Dual contacts and learned ranking | 70.20 | 54.01 | 61.054 | 82.72 |

Values are percentages. H is the harmonic mean of Seen and Unseen. A prediction counts as correct when its rotated IoU is greater than 0.25 and its angle error is less than 30 degrees for at least one annotated grasp matched to the instruction. The reported run uses a project-specific scene-disjoint split, with 40,000 scenes in each test group. [Reported results](reported_results.json) records the instruction counts, split checksums, and aggregate metrics.

## Run the experiments

Use Python 3.10 or newer. A CUDA-enabled PyTorch installation and an NVIDIA GPU are recommended for training and grounding the full corpus. Select a [PyTorch build](https://pytorch.org/get-started/locally/) compatible with your GPU before installing the remaining requirements.

~~~sh
python -m pip install -r requirements.txt
~~~

### Prepare inputs

Get the RGB images and scene descriptions from [Grasp-Anything](https://huggingface.co/datasets/airvlab/Grasp-Anything/tree/6b4917455d0910c23f936a6c7a968418ba518897), and the positive boxes and prompts from [Grasp-Anything++](https://huggingface.co/datasets/airvlab/Grasp-Anything-pp/tree/9b30c841d4f04e4a62e9f7abb3b503337a6200a7). Put the extracted RGB images in data/images. Keep the scene-description, instruction, and positive-label archives as ZIP files.

~~~sh
hf download IDEA-Research/grounding-dino-tiny \
  --revision a2bb814dd30d776dcf7e30523b00659f4f141c71 --local-dir data/grounder
~~~

The four split files should be train.jsonl, val.jsonl, test_seen.jsonl, and test_unseen.jsonl. Each row identifies one instruction and includes scene_id, target_id, image, prompt, and a nonempty grasps list of [cx,cy,w,h,theta] entries in original image coordinates with the angle in radians. Place the split files in data/splits or create fresh scene-disjoint splits as follows. Supply the corresponding archive as --scene-descriptions if its name differs locally.

~~~sh
python common/prepare_full_split.py inventory --work work/inventory \
  --images data/images --scene-descriptions data/scene_descriptions.zip \
  --instructions-zip data/grasp_instructions.zip --positive-zip data/grasp_label_positive.zip \
  --taxonomy 12_full_scale/assets/lvis_v1_categories.json
python common/prepare_full_split.py crosscheck --work work/inventory
python common/prepare_full_split.py rebalance --work work/inventory
python common/prepare_full_split.py plan --work work/inventory
python common/prepare_full_split.py materialize --work work/inventory --out data/splits --workers 4
~~~

The reported benchmark also excluded scenes used in earlier experiments. A fresh split without those inputs has different membership. The splitter accepts prior membership through --history-dir, --prior-used, and --protect on both rebalance and plan. The published split checksums identify an exact match to the benchmark results.

### Train, ground, and score

~~~sh
python common/train_full.py --method 01_film_dense/model.py --images data/images \
  --splits data/splits --output 01_film_dense/full_v1 --skip-test
python common/train_full.py --method 08_contact_pairs/model.py --images data/images \
  --splits data/splits --output 08_contact_pairs/full_v1 --skip-test

python 12_full_scale/panels.py prepare --splits data/splits --output work/panels \
  --all-train --val-count 8192 --shard-size 2048
python 12_full_scale/panels.py cache --panels work/panels --images data/images \
  --model data/grounder --output work/cache
python 12_full_scale/run_2d.py fit --root . --tag full_v1 --panels work/panels \
  --cache work/cache --images data/images --output work/fit
python 12_full_scale/run_2d.py evaluate --root . --tag full_v1 --fit work/fit --panels work/panels \
  --cache work/cache --images data/images --output work/evaluation
python 12_full_scale/run_2d.py summarize --panels work/panels --fit work/fit \
  --evaluation work/evaluation --output work/metrics.json
~~~

The cache and evaluation stages accept --workers N with one job for each --worker index from 0 to N-1. Keep their input manifests fixed when resuming. Candidate heads are fitted on the training split and selected on validation. Test images and grasp annotations are used to measure results after model selection.

## Sources

- [Language-driven Grasp Detection](https://openaccess.thecvf.com/content/CVPR2024/html/Vuong_Language-driven_Grasp_Detection_CVPR_2024_paper.html), CVPR 2024.
- [Grounding DINO](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/6319_ECCV_2024_paper.php), ECCV 2024.
- [Grasp detection and segmentation with a refinement head](https://doi.org/10.1109/ICRA48506.2021.9561398), ICRA 2021.
- [GKNet](https://doi.org/10.1177/02783649211069569), IJRR 2022.
- [LVIS category metadata](12_full_scale/assets/lvis_v1_source.json) is sourced from Detectron2.
