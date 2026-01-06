# MADation: Face Morphing Attack Detection with Foundation Models

This repository contains the official implementation of the paper **"[MADation: Face Morphing Attack Detection with Foundation Models](https://openaccess.thecvf.com/content/WACV2025W/MAPA/papers/Caldeira_MADation_Face_Morphing_Attack_Detection_with_Foundation_Models_WACVW_2025_paper.pdf)"**, accepted at WACV2025 Workshops.

## Overview
Despite significant advancements in face recognition algorithms, morphing attacks pose serious threats. MADation leverages foundation models, specifically the CLIP architecture, adapted with LoRA weights, to effectively detect morphing attacks. Our framework achieves competitive results against existing MAD solutions and is released to encourage reproducibility and further research.

<strong>Features:</strong>
- Foundation model adaptation with LoRA for the MAD task.
- Competitive performance on diverse datasets.
- Comprehensive training and evaluation pipelines for reproducibility.

![Complete pipeline of MADation](/img/pipeline.jpg)  
*Figure 1: Morphing attack generation and MADation's pipeline. The left side of the figure depicts a morphing sample and the two bona-fide identities that were morphed to generate it. Keep in mind that attackers commonly choose to morph faces with similar features for higher success. The right side represents MADation's pipeline, consisting of an adapted FM followed by a binary fully connected classification layer. The embedding space of the FM is adapted by fine-tuning the LoRA parameters and the classification layer is simultaneously trained to produce the MAD predictions. Better visualized in colour.*

![Integration of LoRA trainable weights](/img/mha_lora.jpg)  
*Figure 2: Integration of LoRA trainable weights (orange boxes) in a standard multi-head self-attention block, whose weights are kept frozen (blue boxes). In the proposed framework, MADation, the LoRA adaptation is limited to the q and v matrices, leaving k and o unaltered. Better visualized in colour.*

## How to replicate

- Create a virtual environment by using **requirements.txt**
```
conda create -n env_name python=3.9
pip install -r requirements.txt
```
- Adjust config file in  **/src/config.py**
- Start training with  **./train.sh**

## Pre-trained Models and Training Logs

All pre-trained models and their respective training logs are available **[here](https://drive.google.com/drive/folders/1FAZKgy7Bu_MLJh7UIQFtoMUR0aARUP8v?usp=drive_link)**. To get access, please share your name, affiliation, and email in the request form.

## Key Results

| Method |                 | Test data | EER (%) | APCER (%) @ BPCER (%) |       |       | BPCER (%) @ APCER (%) |       |       |
|:------:|:---------------:|:---------:|:-------:|:---------------------:|:-----:|:-----:|:---------------------:|:-----:|:-----:|
|        |                 |           |         |          1.00         | 10.00 | 20.00 |          1.00         | 10.00 | 20.00 |
|  ViT-B |        TI       | FaceMorph |  51.50  |         98.40         | 88.20 | 81.40 |         99.51         | 93.63 | 85.29 |
|        |                 |  MIPGAN I |  36.40  |         99.80         | 81.10 | 65.30 |         86.76         | 55.88 | 46.57 |
|        |                 | MIPGAN II |  33.40  |         99.60         | 76.00 | 55.30 |         80.39         | 49.02 | 43.63 |
|        |                 |   OpenCV  |  47.15  |         99.90         | 83.74 | 74.90 |         98.04         | 81.37 | 70.10 |
|        |                 |  WebMorph |  35.60  |         98.20         | 70.20 | 57.20 |         86.76         | 61.27 | 48.53 |
|        |                 |  MorDIFF  |  51.90  |         100.00        | 92.60 | 86.70 |         99.02         | 92.65 | 85.29 |
|        |                 |  Average  |  42.66  |         99.32         | 81.97 | 70.13 |         91.75         | 72.30 | 63.24 |
|        |                 |   Worst   |  51.90  |         100.00        | 92.60 | 86.70 |         99.51         | 93.63 | 85.29 |
|        |      ViT-FS     | FaceMorph |   5.38  |          8.77         |  2.49 |  0.90 |         20.98         |  0.49 |  0.00 |
|        |                 |  MIPGAN I |  32.87  |         85.66         | 61.35 | 47.41 |         100.00        | 49.02 | 49.02 |
|        |                 | MIPGAN II |  27.19  |         94.92         | 64.94 | 44.42 |         100.00        | 57.84 | 30.88 |
|        |                 |   OpenCV  |  16.30  |         50.40         | 26.42 | 14.27 |         100.00        | 56.31 | 34.47 |
|        |                 |  WebMorph |  22.80  |         83.60         | 58.00 | 44.40 |         100.00        | 52.94 | 32.35 |
|        |                 |  MorDIFF  |  28.14  |         84.73         | 52.00 | 35.93 |         100.00        | 56.31 | 34.37 |
|        |                 |  Average  |  22.13  |         68.01         | 44.20 | 31.22 |         86.83         | 40.68 | 26.41 |
|        |                 |   Worst   |  32.87  |         94.92         | 64.94 | 47.41 |         100.00        | 57.84 | 49.02 |
|        |        FE       | FaceMorph |   2.89  |          4.89         |  1.30 |  0.20 |         11.22         |  0.49 |  0.49 |
|        |                 |  MIPGAN I |  26.00  |         83.27         | 55.68 | 36.06 |         77.94         | 50.98 | 32.84 |
|        |                 | MIPGAN II |  34.26  |         91.43         | 74.70 | 57.27 |         84.80         | 65.20 | 51.96 |
|        |                 |   OpenCV  |  14.88  |         39.98         | 20.34 |  9.21 |         61.27         | 18.63 | 10.78 |
|        |                 |  WebMorph |  32.80  |         91.40         | 71.40 | 49.80 |         84.80         | 66.18 | 52.94 |
|        |                 |  MorDIFF  |  17.86  |         50.90         | 27.05 | 13.77 |         59.22         | 24.27 | 12.62 |
|        |                 |  Average  |  21.45  |         60.31         | 41.74 | 27.72 |         63.21         | 37.62 | 26.94 |
|        |                 |   Worst   |  34.26  |         91.43         | 74.70 | 57.27 |         84.80         | 66.18 | 52.94 |
|        | MADation (ours) | FaceMorph |   0.00  |          0.00         |  0.00 |  0.00 |          0.00         |  0.00 |  0.00 |
|        |                 |  MIPGAN I |  33.37  |         82.97         | 55.18 | 43.92 |         94.12         | 72.55 | 52.94 |
|        |                 | MIPGAN II |  22.21  |         79.98         | 34.66 | 24.30 |         84.80         | 47.55 | 26.47 |
|        |                 |   OpenCV  |   3.85  |         11.64         |  1.82 |  1.11 |         23.53         |  0.98 |  0.00 |
|        |                 |  WebMorph |  10.80  |         60.00         | 11.40 |  5.00 |         51.47         | 11.76 |  4.41 |
|        |                 |  MorDIFF  |   1.10  |          1.60         |  0.00 |  0.00 |          1.94         |  0.00 |  0.00 |
|        |                 |  Average  |  11.89  |         39.36         | 17.18 | 12.39 |         42.64         | 22.14 | 13.97 |
|        |                 |   Worst   |  33.37  |         82.97         | 55.18 | 43.92 |         94.12         | 72.55 | 52.94 |
|  ViT-L |        TI       | FaceMorph |  44.60  |         98.40         | 79.70 | 63.60 |         99.02         | 87.25 | 76.96 |
|        |                 |  MIPGAN I |  18.90  |         71.80         | 32.20 | 17.80 |         69.61         | 33.82 | 18.14 |
|        |                 | MIPGAN II |  12.80  |         56.70         | 17.00 |  8.90 |         59.31         | 17.16 |  8.33 |
|        |                 |   OpenCV  |  35.47  |         96.24         | 77.54 | 63.11 |         96.08         | 73.53 | 55.39 |
|        |                 |  WebMorph |  25.20  |         94.80         | 52.00 | 30.20 |         87.75         | 50.98 | 32.35 |
|        |                 |  MorDIFF  |  42.60  |         97.80         | 79.60 | 69.50 |         97.06         | 83.33 | 68.63 |
|        |                 |  Average  |  29.93  |         85.96         | 56.34 | 42.19 |         84.81         | 57.68 | 43.30 |
|        |                 |   Worst   |  44.60  |         98.40         | 79.70 | 69.50 |         99.02         | 87.25 | 76.96 |
|        |      ViT-FS     | FaceMorph |  22.63  |         75.17         | 38.68 | 24.93 |         88.29         | 40.98 | 24.88 |
|        |                 |  MIPGAN I |  23.80  |         79.08         | 42.93 | 25.50 |         91.18         | 46.57 | 28.43 |
|        |                 | MIPGAN II |  21.81  |         80.28         | 36.65 | 25.40 |         91.67         | 25.00 | 40.69 |
|        |                 |   OpenCV  |  30.47  |         84.72         | 59.92 | 44.23 |         94.12         | 60.29 | 42.16 |
|        |                 |  WebMorph |  33.60  |         91.60         | 59.80 | 48.60 |         100.00        | 75.49 | 52.45 |
|        |                 |  MorDIFF  |  40.92  |         94.51         | 77.94 | 67.86 |         100.00        | 81.55 | 67.96 |
|        |                 |  Average  |  28.87  |         84.23         | 52.65 | 39.42 |         94.21         | 57.59 | 40.15 |
|        |                 |   Worst   |  40.92  |         94.51         | 77.94 | 67.86 |         100.00        | 81.55 | 67.96 |
|        |        FE       | FaceMorph |   9.77  |         44.17         |  9.77 |  4.09 |         35.12         | 10.24 |  5.37 |
|        |                 |  MIPGAN I |  23.51  |         88.84         | 55.28 | 31.37 |         71.57         | 40.69 | 27.45 |
|        |                 | MIPGAN II |  21.81  |         82.37         | 45.42 | 25.10 |         69.61         | 32.84 | 23.53 |
|        |                 |   OpenCV  |  15.89  |         55.77         | 25.40 | 10.83 |         48.53         | 22.06 | 12.75 |
|        |                 |  WebMorph |  26.40  |         86.60         | 56.80 | 37.80 |         68.63         | 41.67 | 29.90 |
|        |                 |  MorDIFF  |  22.85  |         87.03         | 50.70 | 29.14 |         67.48         | 35.92 | 24.27 |
|        |                 |  Average  |  20.04  |         74.13         | 40.56 | 23.06 |         60.16         | 30.57 | 20.54 |
|        |                 |   Worst   |  26.40  |         88.84         | 56.80 | 37.80 |         71.57         | 41.67 | 29.90 |
|        | MADation (ours) | FaceMorph |   0.40  |          0.40         |  0.00 |  0.00 |          0.49         |  0.00 |  0.00 |
|        |                 |  MIPGAN I |  20.32  |         55.88         | 29.08 | 20.32 |         79.41         | 35.78 | 15.69 |
|        |                 | MIPGAN II |   9.06  |         19.42         |  9.06 |  5.58 |         100.00        |  5.39 |  0.98 |
|        |                 |   OpenCV  |   2.23  |          3.74         |  1.32 |  0.71 |         15.69         |  0.00 |  0.00 |
|        |                 |  WebMorph |  20.40  |         47.60         | 20.40 | 20.40 |         82.35         | 37.25 | 13.24 |
|        |                 |  MorDIFF  |  19.26  |         48.40         | 24.45 | 19.26 |         84.47         | 34.95 | 15.53 |
|        |                 |  Average  |  11.94  |         29.24         | 14.05 | 11.04 |         60.40         | 18.90 |  7.57 |
|        |                 |   Worst   |  20.40  |         55.88         | 29.08 | 20.40 |         100.00        | 37.25 | 15.69 |

Please see the evaluation protocol in **[SYN-MAD-2022](https://github.com/marcohuber/SYN-MAD-2022)**.

## Citation
```
@inproceedings{DBLP:conf/wacv/CaldeiraOCIPBSD25,
  author       = {Eduarda Caldeira and
                  Guray Ozgur and
                  Tahar Chettaoui and
                  Marija Ivanovska and
                  Peter Peer and
                  Fadi Boutros and
                  Vitomir Struc and
                  Naser Damer},
  title        = {MADation: Face Morphing Attack Detection with Foundation Models},
  booktitle    = {{IEEE/CVF} Winter Conference on Applications of Computer Vision, {WACV}
                  2025 - Workshops, Tucson, AZ, USA, February 28 - March 4, 2025},
  pages        = {1565--1575},
  publisher    = {{IEEE}},
  year         = {2025},
  url          = {https://doi.org/10.1109/WACVW65960.2025.00179},
  doi          = {10.1109/WACVW65960.2025.00179},
  timestamp    = {Sat, 06 Sep 2025 20:33:57 +0200},
  biburl       = {https://dblp.org/rec/conf/wacv/CaldeiraOCIPBSD25.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

## License
>This project is licensed under the terms of the **Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)** license.  
Copyright (c) 2025 Fraunhofer Institute for Computer Graphics Research IGD Darmstadt  
For more details, please take a look at the [LICENSE](./LICENSE) file.
