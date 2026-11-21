# FashionVision AI: Requirements Specification

## 1. Industry Use Case

**Domain:** Fashion technology (e-commerce cataloging)

**Problem:** Online fashion retailers list thousands of new garments every
week. Staff manually tag each image with attributes (category, color,
pattern, sleeve type) and write catalog metadata. This is slow,
inconsistent, and expensive.

**Solution:** FashionVision AI automatically:
1. Classifies garment attributes from an image
2. Segments the garment from the background
3. Detects individual items in outfit photos
4. Generates catalog metadata text
5. Generates style variants of a garment
6. Retrieves visually similar items

## 2. Stakeholders and Needs

| Stakeholder | Need | Measurable requirement |
|---|---|---|
| Catalog manager | Fewer tagging errors | Macro F1 >= 0.85 on attributes |
| Merchandiser | Find similar items fast | Retrieval top-5 accuracy >= 0.80 |
| E-commerce developer | Clean product cutouts | Segmentation mean IoU >= 0.70 |
| Stylist / shopper | Multi-item outfit photos | Detection mAP@0.5 >= 0.60 |
| Operations | Real-time throughput | Latency <= 100 ms per image on CPU |
| Legal / ethics | Responsible use | Documented dataset licenses and limits |

*Note: these are target values. Revise them after the first baseline
results, and record the final achieved numbers in the evaluation report.*

## 3. Functional Requirements

| ID | Requirement |
|---|---|
| FR1 | Load an image and predict garment attributes |
| FR2 | Output a segmentation mask for the garment |
| FR3 | Detect and label multiple items with bounding boxes |
| FR4 | Generate a text metadata record for each item |
| FR5 | Generate latent style variants of a garment |
| FR6 | Retrieve the top-k most similar catalog items |
| FR7 | Provide a command-line interface for all functions |
| FR8 | Save results (JSON, images) to the `results/` folder |

## 4. Non-Functional Requirements

| ID | Requirement |
|---|---|
| NFR1 | Builds with CMake on Windows with a single command |
| NFR2 | Runs on CPU only (no GPU required) |
| NFR3 | Memory use under 2 GB during inference |
| NFR4 | Fixed random seeds, so experiments are reproducible |
| NFR5 | Unit tests run with `ctest` and pass |
| NFR6 | Code is modular, with logging and config files |

## 5. Acceptance Tests

| ID | Test | Pass condition |
|---|---|---|
| AT1 | Build from clean clone | Exit code 0, no errors |
| AT2 | Tensor operations | All unit tests pass |
| AT3 | Gradient check | Relative error < 1e-5 |
| AT4 | MLP baseline on Fashion-MNIST | Test accuracy >= 85% |
| AT5 | Optimizer comparison | GD, SGD, Momentum, Nesterov results saved |
| AT6 | Classification | Macro F1 >= 0.85 |
| AT7 | Segmentation | Mean IoU >= 0.70 |
| AT8 | Detection | mAP@0.5 >= 0.60 |
| AT9 | Retrieval | Top-5 accuracy >= 0.80 |
| AT10 | Latency | Median <= 100 ms per image on CPU |
| AT11 | Robustness | F1 drop < 10% under blur, noise, brightness change |
| AT12 | Reproducibility | Same seed gives same results |

## 6. Deployment Constraints

- **Hardware:** standard laptop CPU, 8 GB RAM
- **OS:** Windows 10/11
- **Language:** C++17, built with CMake
- **Dependencies:** minimal and documented in the README
- **Inference:** offline, with no cloud calls

## 7. Data Requirements (summary)

- Source datasets with documented licenses (details in the dataset card)
- Duplicates removed before splitting
- Train / validation / test split of 70 / 15 / 15
- Leakage-safe: the same product or outfit never appears in more than
  one split

## 8. Risks and Responsible-Use Limits

| Risk | Mitigation |
|---|---|
| Dataset bias (body type, skin tone, region) | Report per-group errors |
| Wrong attributes shown to customers | Human review for low-confidence cases |
| Copyright of images | Use only licensed or open datasets |
| Misuse of generated images | Label generated images clearly |

## 9. Go / No-Go Criteria

Recommend deployment only if AT6 to AT10 pass and the robustness test
(AT11) passes. Otherwise, recommend a human-in-the-loop pilot only.