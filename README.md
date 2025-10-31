# FAGR: Feature-Action Generative Replay for Robotic Lifelong Imitation Learning


## Abstract

Conventional imitation learning (IL) struggles with scalability and catastrophic forgetting in sequential task learning, prompting the need for Lifelong Imitation Learning (LIL) to enable sustainable knowledge accumulation. However, existing LIL approaches, which largely depend on replaying demonstration data or generating visual pseudo-samples, suffer from high storage demands and low sample fidelity.

To address these issues, we propose **Feature-Action Generative Replay (FAGR)**, a novel LIL framework for robotic manipulation. FAGR begins by performing multimodal feature extraction and feature-action joint clustering for each task to retain cluster statistics, thereby dramatically reducing storage requirements by replacing raw demonstration data with compact cluster statistics. Subsequently, FAGR generates pseudo-feature samples from these statistics and infers corresponding actions via previously learned policy network, thereby obtaining feature-action pairs for past tasks. To enhance the reliability of these samples, FAGR incorporates a filtering mechanism based on action distribution confidence, which selectively retains the high-fidelity generative samples. These curated samples are then integrated with current-task data for joint optimization, enabling effective robust knowledge retention and transfer. 

To the best of our knowledge, **this work is the first to explore generative replay in the feature-action space for LIL**. Extensive experiments demonstrate that FAGR outperforms state-of-the-art methods in mitigating forgetting while maintaining high task success rates, thus achieving better stability-plasticity balance.

---

## 🚀 Quick Start

**Implementation documentation will come soon.**

---

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@article{yang2025fagr,
  title={FAGR: Feature-Action Generative Replay for Robotic Lifelong Imitation Learning},
  author={Yang, Yushi and Nie, Xiangli and Liu, Chang},
  journal={IEEE Robotics and Automation Letters},
  year={2025}
}
```

---

## 📧 Contact

For questions or collaborations, please contact:
- **Yushi Yang**: yangyushi2023@ia.ac.cn

---
# License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

# Acknowledgements
1. We would like to thank the authors of [LIBERO](https://lifelong-robot-learning.github.io/LIBERO/) for providing the datasets, environments and codebase for our experiments.
