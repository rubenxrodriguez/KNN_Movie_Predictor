# Machine Learning Final Exam Study Guide: kNN and Clustering Concepts

## 1. k-Nearest Neighbors (kNN) Algorithm

### 1.1 Core Concepts
- Distance Metrics:
  * Role in kNN: Determines similarity between instances
  * Common metrics: Euclidean, Manhattan, Minkowski, Cosine
  * Impact on neighbor selection and algorithm performance

- Choosing k Value:
  * Bias-variance tradeoff (small k vs large k)
  * Sensitivity to noise (small k overfits, large k underfits)
  * Computational cost implications
  * Odd vs even k in binary classification

### 1.2 Practical Considerations
- Class Imbalance:
  * Majority class dominance in voting
  * Potential solutions: weighted voting, sampling techniques

- Feature Scaling:
  * Why necessary: distance-based algorithms are scale-sensitive
  * Common methods: Min-Max, Standardization
  * Consequences of not scaling

- Regression Adaptation:
  * Average/weighted average of neighbors' values
  * Distance-weighted predictions

### 1.3 Advanced Applications
- Recommendation Systems:
  * User/item similarity measurement
  * Collaborative filtering approaches
  * Handling sparsity in data

### 1.4 Comparative Analysis
- vs Model-based Algorithms:
  * Lazy vs eager learning
  * Interpretability differences
  * Training vs prediction time complexity
  * Handling of feature relationships

### 1.5 Strengths and Limitations
- Advantages:
  * Simple implementation
  * No training phase
  * Naturally handles multi-class problems
- Limitations:
  * Computational cost at prediction time
  * Curse of dimensionality
  * Sensitive to irrelevant features

## 2. Clustering Algorithms

### 2.1 Fundamental Concepts
- Learning Context:
  * Unsupervised learning paradigm
  * Discovery of inherent groupings

- Business Applications:
  * Customer segmentation
  * Anomaly detection
  * Data preprocessing for other algorithms

### 2.2 Algorithm Comparison
- K-Means vs Hierarchical:
  * Partitioning vs hierarchical approach
  * Complexity differences
  * Cluster shape assumptions
  * Deterministic vs probabilistic outcomes

### 2.3 High-Dimensional Challenges
- Curse of dimensionality:
  * Distance concentration problems
  * Dimensionality reduction solutions (PCA, t-SNE)
  * Alternative metrics (cosine similarity)

### 2.4 Cluster Evaluation
- Quality Assessment:
  * Internal metrics (cohesion, separation)
  * External metrics (when labels available)
  * Visual inspection methods

- Determining Optimal k:
  * Elbow Method: SSE analysis
  * Silhouette Score: cohesion vs separation
  * Gap statistics
  * Business context considerations

### 2.5 Real-World Applications
- Business Strategy Example:
  * Retail customer segmentation
  * Resource allocation based on clusters
  * Targeted marketing strategies

### 2.6 K-Means Characteristics
- Advantages:
  * Scalability to large datasets
  * Simple interpretation
  * Fast convergence for well-separated data
- Limitations:
  * Spherical cluster assumption
  * Sensitivity to initialization
  * Difficulty with varying density clusters

## 3. Comparative Summary
- kNN vs Clustering:
  * Supervised vs unsupervised contexts
  * Different use cases (prediction vs pattern discovery)
  * Similar sensitivity to distance metrics and scaling
