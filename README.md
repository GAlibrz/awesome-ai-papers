# 📚 AI Papers I’m Reading

A curated list of papers I’ve read (or plan to read) in **AI, LLMs, Reinforcement Learning, Federated Learning, Agentic AI, Computer Vision, and Meta Learning**, with short notes for context.  
This serves as my personal learning log and a reference for others interested in applied AI research.  

---

## 🔹 Foundations
- [Attention is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)  
  → Introduced the Transformer architecture, foundation of modern NLP and LLMs.  
- [Deep Residual Learning for Image Recognition (He et al., 2015)](https://arxiv.org/abs/1512.03385)  
  → ResNet, enabling very deep networks with skip connections.  
- [Adam: A Method for Stochastic Optimization (Kingma & Ba, 2015)](https://arxiv.org/abs/1412.6980)  
  → Optimizer widely used for training neural networks.  

---

## 🔹 Computer Vision
- [ImageNet Classification with Deep Convolutional Neural Networks (AlexNet, Krizhevsky et al., 2012)](https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)  
  → Sparked the deep learning revolution in vision, won ImageNet 2012 by a large margin.

- [Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG, Simonyan & Zisserman, 2014)](https://arxiv.org/abs/1409.1556)  
  → Showed the power of depth using small 3×3 filters; widely used in transfer learning.

- [Deep Residual Learning for Image Recognition (ResNet, He et al., 2015)](https://arxiv.org/abs/1512.03385)  
  → Introduced residual connections, enabling very deep networks (50–1000+ layers).

- [Going Deeper with Convolutions (Inception/GoogLeNet, Szegedy et al., 2015)](https://arxiv.org/abs/1409.4842)  
  → Multi-path convolutional modules; efficient and accurate architecture.

- [Densely Connected Convolutional Networks (DenseNet, Huang et al., 2017)](https://arxiv.org/abs/1608.06993)  
  → Feature reuse via dense connections; fewer parameters with competitive accuracy.

- [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters (Iandola et al., 2016)](https://arxiv.org/abs/1602.07360)  
  → Lightweight CNN architecture optimized for embedded systems and efficiency.

- [You Only Look Once: Unified, Real-Time Object Detection (YOLO, Redmon et al., 2016)](https://arxiv.org/abs/1506.02640)  
  → Real-time object detection with a single forward pass; fast and simple.

- [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks (Ren et al., 2015)](https://arxiv.org/abs/1506.01497)  
  → Two-stage detection pipeline using CNNs for region proposals and classification.

- [Mask R-CNN (He et al., 2017)](https://arxiv.org/abs/1703.06870)  
  → Extended Faster R-CNN for instance segmentation; highly influential in detection and segmentation.

- [Feature Pyramid Networks for Object Detection (FPN, Lin et al., 2017)](https://arxiv.org/abs/1612.03144)  
  → Multi-scale feature aggregation; improved detection for small objects.

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT, Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929)  
  → Applied Transformer architecture directly to image patches; opened new direction in vision.

- [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows (Liu et al., 2021)](https://arxiv.org/abs/2103.14030)  
  → Introduced local windows and hierarchical structure to ViT; highly scalable and strong results.

- [Masked Autoencoders Are Scalable Vision Learners (MAE, He et al., 2021)](https://arxiv.org/abs/2111.06377)  
  → Pretrained ViTs via masked autoencoding; efficient and effective self-supervised learning.

- [Segment Anything (Kirillov et al., 2023)](https://arxiv.org/abs/2304.02643)  
  → Introduced a foundation model for promptable segmentation across arbitrary image domains.

- [DINO: Self-Distillation with no Labels (Caron et al., 2021)](https://arxiv.org/abs/2104.14294)  
  → Self-supervised vision representation learning using teacher–student ViTs.

- [SimCLR: A Simple Framework for Contrastive Learning (Chen et al., 2020)](https://arxiv.org/abs/2002.05709)  
  → Landmark paper in contrastive self-supervised learning; strong results without labels.

---

## 🔹 Large Language Models
- [BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)](https://arxiv.org/abs/1810.04805)  
  → Introduced bidirectional transformer pretraining, foundational for NLP.  
- [Retrieval-Augmented Generation (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)  
  → RAG, combining retrieval with generation for enriched context.  
- [AutoGen: Enabling Next-Gen LLM Applications (Microsoft, 2023)](https://arxiv.org/abs/2308.08155)  
  → Multi-agent orchestration with LLMs.  

---

## 🔹 Reinforcement Learning
- [Human-level control through deep reinforcement learning (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)  
  → Deep Q-Network (DQN), combining deep learning with RL.  
- [Proximal Policy Optimization (PPO) (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)  
  → Simple, stable, and efficient policy optimization.  
- [REINFORCE: Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning (Williams, 1992)](https://www.jmlr.org/papers/volume4/williams02a/williams02a.pdf)  
  → Introduced REINFORCE algorithm, the foundation of policy gradient methods.
- [Trust Region Policy Optimization (TRPO, Schulman et al., 2015)](https://arxiv.org/abs/1502.05477)  
  → Introduced KL-constrained updates for stable policy learning, precursor to PPO.
- [Generalized Advantage Estimation (GAE, Schulman et al., 2015)](https://arxiv.org/abs/1506.02438)  
  → Variance reduction technique for actor-critic methods, widely used with PPO.

- [Actor-Critic Algorithms (Konda & Tsitsiklis, 1999)](https://proceedings.neurips.cc/paper_files/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)  
  → Theoretical foundation of the actor-critic architecture in policy-based RL.
---

## 🔹 Federated Learning
- [Communication-Efficient Learning of Deep Networks from Decentralized Data (McMahan et al., 2017)](https://arxiv.org/abs/1602.05629)  
  → FedAvg, foundational algorithm in federated learning.  
- [Advances and Open Problems in Federated Learning (Kairouz et al., 2019)](https://arxiv.org/abs/1912.04977)  
  → Comprehensive survey of challenges and directions.  

---

## 🔹 Agentic AI & Multi-Agent Systems
- [Voyager: An Open-Ended Embodied Agent (Wang et al., 2023)](https://arxiv.org/abs/2305.16291)  
  → LLM-powered agent for continual learning in open worlds.  
- [AutoGen: Enabling Next-Gen LLM Applications (Microsoft, 2023)](https://arxiv.org/abs/2308.08155)  
  → Framework for multi-agent orchestration of LLMs.  

---

## 🔹 Meta-Learning
- [Learning to Learn by Gradient Descent by Gradient Descent (Andrychowicz et al., 2016)](https://arxiv.org/abs/1606.04474)  
  → Introduced meta-learning where an optimizer itself is learned through gradient descent.  

## 📌 Goal
This repository is my personal learning log to stay engaged with key AI research.  
Notes and insights will be added as I progress. Contributions and suggestions are welcome!