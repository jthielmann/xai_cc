# Todo List

## Task 1: Train ResNet50 v2

1.  **Update Configuration**:
    *   Modify your training configuration file (e.g., a YAML or Python config).
    *   Set the `encoder_type` parameter to `"resnet50imagenet"`. Based on `script/model/model_factory.py`, this will load a ResNet50 with ImageNet V2 weights.

2.  **Start Training**:
    *   Run your main training script, likely `script/main.py`, pointing it to the updated configuration file.
    *   Example: `python script/main.py --config /path/to/your/config.yml`

3.  **Monitor and Evaluate**:
    *   Track the training progress using your logger (WandB or TensorBoard).
    *   After training, find the results and model checkpoints in the `out_path` directory specified in your config. The performance metrics will be in the `results/` directory as outlined in `script/model/lit_model.py`.

## Task 2: Train SAE per Spatial Location

The current SAE implementation in `lit_model.py` and `lit_train_sae.py` applies the autoencoder to the flattened and pooled feature vector from the encoder. To train an SAE on the features at each spatial location (h, w), you will need to make some modifications.

1.  **Modify the Model's Forward Pass**:
    *   In `script/model/lit_model.py`, locate the `forward` method.
    *   Currently, it reshapes 4D tensors `(B, C, H, W)` to 2D `(B, D)`.
    *   You need to change this logic. Before the SAE is applied, reshape the encoder output `z` from `(B, C, H, W)` to `(B * H * W, C)`.
    *   After the SAE, you may need to reshape it back to `(B, C, H, W)` or `(B, D)` depending on what the downstream gene expression heads expect.

2.  **Update SAE Configuration**:
    *   The `d_in` for the `SparseAutoencoder` in `script/model/lit_ae.py` will now be the number of channels (`C`) from the encoder output, not the flattened dimension. Ensure your configuration reflects this.

3.  **Create a Dedicated Training Script**:
    *   Copy `script/train/lit_train_sae.py` to a new file, e.g., `script/train/lit_train_sae_spatial.py`.
    *   In this new script, adjust the `SAETrainerPipeline` to implement the new reshaping logic. The `setup` and `run` methods will need to be adapted to handle the `(B*H*W, C)` input to the SAE.

4.  **Configure and Run**:
    *   Create a new configuration file for this experiment.
    *   Run your new training script `lit_train_sae_spatial.py`.

## Task 3: Verify Gene Distribution in Test Set

The script `script/evaluation/generate_data_hists.py` already provides a good foundation for this. You can adapt it to compare train and test splits.

1.  **Adapt the Histogram Script**:
    *   Create a new script, e.g., `script/evaluation/compare_split_dists.py`, by modifying `generate_data_hists.py`.
    *   In the new script, load both the training and test sets. Your `lit_STDataModule.py` should have methods to provide these splits.
    *   For each gene, generate overlaid histograms to visually compare the distribution of values between the training and test sets.

2.  **Calculate Summary Statistics**:
    *   In the same script, for each gene, calculate and print a comparison table with key statistics (mean, median, standard deviation, min, max) for both the train and test splits.

3.  **Perform Statistical Test (Optional)**:
    *   For a more rigorous comparison, you can use `scipy.stats.ks_2samp` (Kolmogorov-Smirnov test) to get a p-value indicating if the two distributions are significantly different.

4.  **Document Findings**:
    *   Save the generated plots and the summary statistics table to a markdown file or a notebook in the `docs/` directory to document your findings.
