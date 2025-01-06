# MADation: Face Morphing Attack Detection with Foundation Models

This repository contains the official implementation of the paper **"[MADation: Face Morphing Attack Detection with Foundation Models](link-to-paper),"**, accepted at WACV2025 WACVMAPA2025 (Workshop on Manipulation, Generative, Adversarial and Presentation Attacks In Biometrics (MAP-A)).

## Overview
Despite significant advancements in face recognition algorithms, morphing attacks pose serious threats. MADation leverages foundation models, specifically the CLIP architecture, adapted with LoRA weights, to effectively detect morphing attacks. Our framework achieves competitive results against existing MAD solutions and is released to encourage reproducibility and further research.

<strong>Features:</strong>
- Foundation model adaptation with LoRA for the MAD task.
- Competitive performance on diverse datasets.
- Comprehensive training and evaluation pipelines for reproducibility.

![Complete pipeline of MADation](/img/pipeline.jpg)  
*Morphing attack generation and MADation's pipeline. The left side of the figure depicts a morphing sample and the two bona-fide identities that were morphed to generate it. Keep in mind that attackers commonly choose to morph faces with similar features for higher success. The right side represents MADation's pipeline, consisting of an adapted FM followed by a binary fully connected classification layer. The embedding space of the FM is adapted by fine-tuning the LoRA parameters and the classification layer is simultaneously trained to produce the MAD predictions. Better visualized in colour.*

![Integration of LoRA trainable weights](/img/mha_lora.jpg)  
*Integration of LoRA trainable weights (orange boxes) in a standard multi-head self-attention block, whose weights are kept frozen (blue boxes). In the proposed framework, MADation, the LoRA adaptation is limited to the $q$ and $v$ matrices, leaving $k$ and $o$ unaltered. Better visualized in colour.*

## Key Results - ViT-B/16


## Key Results - ViT-L/14






## Citation
```
@inproceedings{madation2025,
  title={MADation: Face Morphing Attack Detection with Foundation Models},
  author={Caldeira, Eduarda and Ozgur, Guray and Chettaoui, Tahar and Ivanovska, Marija and Boutros, Fadi and Struc, Vitomir and Damer, Naser},
  booktitle={WACV WACVMAPA2025 Workshop},
  year={2025},
  institution={Fraunhofer IGD, TU Darmstadt, University of Ljubljana}
}
```

## License
>This project is licensed under the terms of the **Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)** license.  
Copyright (c) 2025 Fraunhofer Institute for Computer Graphics Research IGD Darmstadt  
For more details, please take a look at the [LICENSE](./LICENSE) file.
