#  MAP: Low-compute Model Merging with Amortized Pareto Fronts via Quadratic Approximation
 <div align="center">
  <img src="img/map_logo.jpeg" alt="map Logo" width="200"/>
</div>

## Code

### Quick Start

```bash
conda env create -f environment.yaml
conda activate map
git clone https://github.com/mlfoundations/task_vectors # we used the zero-shot task vectors from this repo
python run_MAP.py # check the args.py for the hyperparameters and 4 modes
```

# Introduction

<p align="center">
<img src="img/main_fig.png" alt="scatter" width="100%"/>
</p>

The overall process of MAP under 2 tasks. 
- Step 1: select tasks and their corresponding task vectors. 
- Step 2: sample a few scaling weights $c$ to query the task metrics accordingly.
- Step 3: use a quadratic model to approximate the mapping of $c \rightarrow \textit{metrics}$ as a surrogate model.
- Step 4: use NSGA-III (or other multi-objective optimization algorithms) to find amortized Pareto fronts.

Figure (a) shows the contour plots of the actual accuracy landscape for the ViT models obtained from 100 scaling coefficients (sampled uniformly) evaluated on the SUN397 dataset and Cars dataset. 
Figure (b) shows the contour plots of the fitted quadratic functions. Red lines represent the Pareto front in the decision variable $(c_1, c_2)$ space. 
Figure (c) shows an example of the Pareto Front. Pareto front (Grid search) is regarded as the ground truth given the enough number of grid points. Pareto front (MAP, predicted) is the amortized Pareto front. Pareto front (MAP, real) is the Pareto front involving the same $\{(c_1,c_2)\}$ but re-evaluated as to get the ground truth metrics as a comparison. The yellow lines are the evaluation performance of the fine-tuned single task models.
