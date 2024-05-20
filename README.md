# Multi-Level GNN Preconditioner for Solving Large Scale Problems

Paper : https://arxiv.org/abs/2402.08296

### Environment Setup

Environment can be configured using conda and the provided env.yml file.

### Generate the dataset

The dataset can be generated in the train/dataset folder by running the launch_dataset.sh file. It is possible to configure the path to save the dataset, the number of global problems and the desired number of points per subdomains. Information about the dataset are provided in the data/ folder.

### Train the model

Run the launch_model.sh file in the train/gnn_model folder. Hyperparameters of the model can be modified in the config.json file. It is possible to choose either "ddp_slurm" for Distributed Data Parallel on a slurm server "ddp_local" for Distributed Data Parallel on local machine or "dp" for DataParallel.

### Evaluation

Evaluation is done in the evaluation/ folder. Run the test_many_models_multiple.py file to evaluate the performance of models with various configurations on several problems of the same size. Return the averaged and std metrics. To evaluate, one model on one geometry, use the test_one_model.py file and to evaluate one model on various problems, run the test_many_prbs.py file.