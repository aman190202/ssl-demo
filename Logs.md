
runs/mae_galaxy10
The MAE pretraining run successfully converged in reconstruction loss, dropping from 0.29 to around 0.08, indicating that the model has learned to fill in masked image regions effectively. However, both the k-NN and linear-probe evaluations suggest that the encoder’s learned features are still weakly separable: validation top-1 accuracy peaked near 55 %, and the linear probe stabilized around 15 %. This gap implies that while the decoder is reconstructing images well, the encoder has not yet developed strong semantic representations. Further tuning—longer training, richer augmentations, and adjusted mask ratio or embedding size—is needed to improve representation quality before downstream classification performance becomes meaningful.

runs/mae_galaxy10_v2
Increased the depth of the encoder to reduce underfitting and reduced the depth of the decoder so that the encoder is forced to learn better features. Reduced the masking ratio because a large portion of the image is dark and masking 75% might make most images look the same. The MAE is converging slowly.

runs/mae_galaxy10_v3
Configuration: enc_depth=12, emb_dim=384, dec_dim=256, mask_ratio=0.65, cosine LR with warmup. Loss dropped from ~1.21 to ~0.053. k-NN val/test top-1 rose to ~0.55 around epochs 10–16, then decayed to ~0.43 by epoch 133 despite continued loss improvements—consistent with objective–metric mismatch (pixel MSE favoring low-frequency reconstructions) and small-data overfit. Reconstructions (“masked_filled”) looked reasonable; “recon_full” remained blurry for the same reason.

runs/mae_galaxy10_v3_linear_probe
With the encoder frozen (correct arch/ckpt loaded), the linear probe improved steadily: val top-1 climbed from ~0.15 to ~0.33 by ~300 epochs; test top-1 = 0.3343. This confirms fixes to checkpoint loading/arch alignment and establishes a stronger baseline for downstream classification than earlier probes.


runs/mae_latest
Test accuracy climbed to 47% 

Linear Probe
attaching linear probe gave a deeper insight into what is happening ; will experiment with different settings now - will mask more parts of the image and make masking not random 