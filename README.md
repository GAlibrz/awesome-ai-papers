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
- [Focal Loss for Dense Object Detection (Lin et al., 2017)](https://arxiv.org/abs/1708.02002)  
  → Introduced focal loss to address class imbalance in dense detection tasks; core of RetinaNet.
- [Understanding the Difficulty of Training Deep Feedforward Neural Networks (Glorot & Bengio, 2010)](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)  
  → Introduced Xavier initialization, addressing vanishing/exploding gradients in deep networks.
- [Long Short-Term Memory (Hochreiter & Schmidhuber, 1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)  
  → Introduced the LSTM architecture to overcome vanishing gradients in RNNs; widely used in sequence modeling before Transformers.
- [Approximation by Superpositions of a Sigmoidal Function (Cybenko, 1989)](https://epubs.siam.org/doi/abs/10.1137/0149056)  
  → Proved the Universal Approximation Theorem: a single hidden layer neural network with non-linear activations can approximate any continuous function.

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
- [Rethinking the Inception Architecture for Computer Vision (Szegedy et al., 2016)](https://arxiv.org/abs/1512.00567)  
  → Introduced label smoothing regularization to prevent overconfidence in classification; improves generalization.
- [ImageNet Classification with Deep Convolutional Neural Networks (AlexNet, Krizhevsky et al., 2012)](https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)  
  → Kicked off the deep learning era by winning ImageNet 2012 with CNNs and ReLU activation.

- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift (Ioffe & Szegedy, 2015)](https://arxiv.org/abs/1502.03167)  
  → Introduced BatchNorm to stabilize and speed up training, now standard in almost all models.

- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting (Srivastava et al., 2014)](https://jmlr.org/papers/v15/srivastava14a.html)  
  → Regularization technique that randomly drops units during training to improve generalization.

- [Rectified Linear Units Improve Restricted Boltzmann Machines (ReLU, Nair & Hinton, 2010)](https://www.cs.toronto.edu/~hinton/absps/reluICML.pdf)  
  → Introduced ReLU activation, replacing sigmoid/tanh and enabling deeper networks.

- [Attention is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)  
  → Introduced the Transformer architecture, foundational for modern deep learning models across modalities.

- [Distilling the Knowledge in a Neural Network (Hinton et al., 2015)](https://arxiv.org/abs/1503.02531)  
  → Introduced knowledge distillation, enabling smaller models to learn from larger teacher models.

---

## 🔹 Large Language Models
- [BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)](https://arxiv.org/abs/1810.04805)  
  → Introduced bidirectional transformer pretraining, foundational for NLP.  
- [Retrieval-Augmented Generation (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)  
  → RAG, combining retrieval with generation for enriched context.  
- [AutoGen: Enabling Next-Gen LLM Applications (Microsoft, 2023)](https://arxiv.org/abs/2308.08155)  
  → Multi-agent orchestration with LLMs.  

- [Why Language Models Hallucinate (OpenAI, 2023)](https://cdn.openai.com/pdf/d04913be-3f6f-4d2b-b283-ff432ef4aaa5/why-language-models-hallucinate.pdf)  
  → Analyzed the root causes of hallucination in LLMs; shows that overly strong priors and misaligned training objectives contribute significantly.

---

## 🔹 LLMs for Log Analysis & Monitoring

- [LogGPT: A Unified Generative Framework for Log Anomaly Detection and Diagnosis (Zhang et al., 2023)](https://arxiv.org/abs/2305.16291)  
  → Uses LLMs for both anomaly detection and root cause analysis in system logs. Introduces a pretraining-finetuning setup on synthetic and real-world logs.

- [LogPrompt: Enhancing Log Anomaly Detection via Few-Shot Prompting with Large Language Models (Shen et al., 2023)](https://arxiv.org/abs/2305.11465)  
  → Shows that few-shot prompting with GPT-style LLMs can effectively identify anomalies in raw logs, outperforming some fine-tuned baselines.

- [LogSummary: Abstractive Summarization of Log Reports with LLMs (Cao et al., 2023)](https://arxiv.org/abs/2308.13969)  
  → Uses LLMs for automatic log summarization, bridging human-readable insights and raw logs for monitoring/debugging.

- [LLM4Logs: Large Language Models are Zero-Shot Detectors for Logs (Sun et al., 2023)](https://arxiv.org/abs/2306.07071)  
  → Proposes a framework where LLMs act as zero-shot classifiers for log sequences; no training required, prompt-only setup.

- [LogGLM: A Unified Generative Framework for Log Understanding with Large Language Models (Zhang et al., 2023)](https://arxiv.org/abs/2310.02225)  
  → Proposes a unified generative model for log anomaly detection, summarization, and classification. Leverages a pretrained GLM backbone with domain adaptation for system log understanding.
---

## 🔹 Reinforcement Learning
- [Human-level control through deep reinforcement learning (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)  
  → Deep Q-Network (DQN), combining deep learning with RL.  
- [Proximal Policy Optimization (PPO) (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)  
  → Simple, stable, and efficient policy optimization.  
- [REINFORCE: Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning (Williams, 1992)](https://www.jmlr.org/papers/volume4/williams02a/williams02a.pdf)  
- [Asynchronous Methods for Deep Reinforcement Learning (Mnih et al., 2016)](https://arxiv.org/abs/1602.01783)  
  → A3C, massively parallel RL training, influential for scalability.  

  → Introduced REINFORCE algorithm, the foundation of policy gradient methods.
- [Trust Region Policy Optimization (TRPO, Schulman et al., 2015)](https://arxiv.org/abs/1502.05477)  
  → Introduced KL-constrained updates for stable policy learning, precursor to PPO.
- [Generalized Advantage Estimation (GAE, Schulman et al., 2015)](https://arxiv.org/abs/1506.02438)  
  → Variance reduction technique for actor-critic methods, widely used with PPO.

- [Actor-Critic Algorithms (Konda & Tsitsiklis, 1999)](https://proceedings.neurips.cc/paper_files/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)  
  → Theoretical foundation of the actor-critic architecture in policy-based RL.

- [Mastering the Game of Go with Deep Neural Networks and Tree Search (Silver et al., 2016)](https://www.nature.com/articles/nature16961)  
  → AlphaGo, milestone in combining deep learning with MCTS.  

- [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm (Silver et al., 2017)](https://arxiv.org/abs/1712.01815)  
  → AlphaZero, general RL system surpassing human champions.  

- [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor (Haarnoja et al., 2018)](https://arxiv.org/abs/1801.01290)  
  → Introduced entropy-regularized RL for better exploration and stability. SAC is a widely used off-policy actor-critic algorithm for continuous control.

- [Decision Transformer: Reinforcement Learning via Sequence Modeling (Chen et al., 2021)](https://arxiv.org/abs/2106.01345)  
  → Reformulates RL as a sequence modeling problem using transformers. Achieves strong results using supervised learning with trajectory return conditioning.
---

## 🔹 Federated Learning
- [Communication-Efficient Learning of Deep Networks from Decentralized Data (McMahan et al., 2017)](https://arxiv.org/abs/1602.05629)  
  → FedAvg, foundational algorithm in federated learning.  
- [Advances and Open Problems in Federated Learning (Kairouz et al., 2019)](https://arxiv.org/abs/1912.04977)  
  → Comprehensive survey of challenges and directions.  

- [Federated Optimization in Heterogeneous Networks (Li et al., 2020)](https://arxiv.org/abs/1812.06127)  
  → Introduced FedProx, an extension of FedAvg designed to handle data and system heterogeneity—one of the biggest challenges in real-world FL.

- [Adaptive Federated Optimization (Reddi et al., 2021)](https://arxiv.org/abs/2003.00295)  
  → Proposed FedOpt, FedAdam, and FedYogi — server-side adaptive optimizers that significantly improve convergence and stability in federated settings.

---

## 🔹 Agentic AI & Multi-Agent Systems

- [ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022)](https://arxiv.org/abs/2210.03629)  
  → Introduced ReAct framework, combining reasoning traces and actions for decision-making agents.

- [Toolformer: Language Models Can Teach Themselves to Use Tools (Schick et al., 2023)](https://arxiv.org/abs/2302.04761)  
  → Self-supervised approach enabling LMs to decide *when and how* to call external tools.

- [AutoGPT (Toran Bruce Richards, 2023)](https://github.com/Torantulino/Auto-GPT)  
  → Popularized autonomous LLM agents that recursively plan and act using tools and memory.

- [Voyager: An Open-Ended Embodied Agent with LLMs (Xu et al., 2023)](https://voyager.minedojo.org/)  
  → Minecraft agent that autonomously explores, learns skills, and builds a curriculum using GPT-4.

- [Plan-and-Solve Prompting: Leveraging Structured Plans for Complex Reasoning (Zhou et al., 2022)](https://arxiv.org/abs/2211.03544)  
  → Uses explicit planning prompts to structure multi-step reasoning with LLMs.

- [AutoAgents: Enabling LLMs to Operate Autonomously (Qin et al., 2023)](https://arxiv.org/abs/2303.16725)  
  → Proposed a framework for multi-agent LLM systems with memory, reflection, and role delegation.

- [BabyAGI (Yohei Nakajima, 2023)](https://github.com/yoheinakajima/babyagi)  
  → A minimalistic autonomous task-management system using an LLM and vector database.

- [Generative Agents: Interactive Simulacra of Human Behavior (Park et al., 2023)](https://arxiv.org/abs/2304.03442)  
  → Simulated believable, memory-augmented agents in a sandbox world; foundation for agent-based simulations.



---

## 🔹 Meta-Learning
- [Learning to Learn by Gradient Descent by Gradient Descent (Andrychowicz et al., 2016)](https://arxiv.org/abs/1606.04474)  
  → Introduced meta-learning where an optimizer itself is learned through gradient descent.  

## 📌 Goal
This repository is my personal learning log to stay engaged with key AI research.  
Notes and insights will be added as I progress. Contributions and suggestions are welcome!