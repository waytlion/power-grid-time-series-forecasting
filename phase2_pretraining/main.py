from gridfm_graphkit.models.gnn_heterogeneous_gns import GNS_heterogeneous
from gridfm_encoder import GridFMEncoder

# 1. Create GridFM model (from config or manually)
gridfm = GNS_heterogeneous(args)

# 2. Wrap with encoder
encoder = GridFMEncoder(gridfm)
encoder.reset_parameters()  # Fresh start for end-to-end training

print(encoder)
# GridFMEncoder(
#   task=StateEstimation,
#   hidden_dim=48,
#   heads=8,
#   output_dim=384,
#   num_layers=12,
#   trainable=True
# )

# 3. Forward pass (embedding only - for inference/TGT input)
out = encoder(x_dict, edge_index_dict, edge_attr_dict, mask_dict)
embeddings = out["embedding"]  # [14, 384] for IEEE-14

# 4. Forward pass with reconstruction (for multi-task training)
out = encoder(x_dict, edge_index_dict, edge_attr_dict, mask_dict, return_reconstruction=True)

loss_forecast = mse(tgt_output, future_target)
loss_recon = mse(out["pred_bus"], target_bus) + mse(out["pred_gen"], target_gen)
total_loss = loss_forecast + lambda_recon * loss_recon