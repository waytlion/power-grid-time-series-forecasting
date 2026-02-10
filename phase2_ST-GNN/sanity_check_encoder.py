# This script checks the forward pass of the GridFMEncoder with a dummy batch. 
# It ensures that the encoder can process the input and produce outputs without errors, 
# and that the output dimensions are as expected. 
# This is a sanity check before integrating the encoder into the full training loop.


import torch
from models.gnn_heterogenous_gns_thesis import GNS_heterogeneous_thesis
from Wrapper import GridFMEncoder

# 1. Mock Configuration (Nested structure to match GNS_heterogeneous expectatons)
class MockModelConfig:
    num_layers = 2
    hidden_size = 16      
    attention_head = 2   
    dropout = 0.0
    
    input_bus_dim = 15    
    input_gen_dim = 15     
    output_bus_dim = 4    # Vm, Va,Pg, Qg 
    output_gen_dim = 1    # Pg 
    edge_dim = 11         # All edge features (P, Q, Yff_r, Yff_i, Yft_r, Yft_i, TAP, ANG_MIN, ANG_MAX, RATE_A, B_ON)

class MockTaskConfig:
    task_name = "StateEstimation"

class MockArgs:
    model = MockModelConfig()
    task = MockTaskConfig()

args = MockArgs()

# 2. Instantiate Model & Encoder
print("Initializing GridFM...")
gridfm = GNS_heterogeneous_thesis(args)
encoder = GridFMEncoder(gridfm)
encoder.reset_parameters()

# 3. Create Dummy Batch (Batch Size = 2, IEEE-14 Topology)
# IEEE-14 has 14 Buses, 5 Generators.
# Batch Size 2 -> 28 Buses, 10 Generators in total (PyG batching).
batch_size = 2
n_bus_per_graph = 14
n_gen_per_graph = 5
n_total_bus = batch_size * n_bus_per_graph
n_total_gen = batch_size * n_gen_per_graph

# Dummy Features (Features dimensions taken from standard GridFM, e.g., 9)
x_dict = {
    "bus": torch.randn(n_total_bus, args.model.input_bus_dim),  # [28, 15]
    "gen": torch.randn(n_total_gen, args.model.input_gen_dim)   # [10, 15]
}

# Dummy Edges (simplified: Gen connected to Bus)
# We map 5 gens to first 5 buses in each graph
gen_indices = torch.arange(n_total_gen)
bus_indices = torch.arange(n_total_gen) # Just connect Gen i to Bus i for simplicity
# Need to offset the second batch
bus_indices[n_gen_per_graph:] += (n_bus_per_graph - n_gen_per_graph) 

edge_index_dict = {
    ("gen", "connected_to", "bus"): torch.stack([gen_indices, bus_indices]),
    ("bus", "connected_to", "gen"): torch.stack([bus_indices, gen_indices]),  # Reverse edge
    ("bus", "connects", "bus"): torch.randint(0, n_total_bus, (2, 40)) # Random connections
}

edge_attr_dict = {
    ("bus", "connects", "bus"): torch.randn(40, 11), # All 11 edge features (P, Q, Yff_r, Yff_i, Yft_r, Yft_i, TAP, ANG_MIN, ANG_MAX, RATE_A, B_ON)
    ("gen", "connected_to", "bus"): None,
    ("bus", "connected_to", "gen"): None  # No edge attributes for this relation

}

# Dummy Masks (50% masked)
mask_dict = {
    "bus": torch.randint(0, 2, (n_total_bus, args.model.input_bus_dim)).bool(),
    "gen": torch.randint(0, 2, (n_total_gen, args.model.input_gen_dim)).bool()
}

# 4. RUN FORWARD PASS
print("\n--- Running Forward Pass (Mode: Inference) ---")
out_inference = encoder(x_dict, edge_index_dict, edge_attr_dict, mask_dict, return_reconstruction=False)
print("Keys in output:", out_inference.keys())
print("Embedding Shape:", out_inference["embedding"].shape)

print("\n--- Running Forward Pass (Mode: Training/Reconstruction) ---")
out_train = encoder(x_dict, edge_index_dict, edge_attr_dict, mask_dict, return_reconstruction=True)
print("Keys in output:", out_train.keys())

print(f"\nExpected Embedding Dim = hidden_size * attention_head: {args.model.hidden_size * args.model.attention_head}")