"""
GridFM Encoder Wrapper for End-to-End Spatio-Temporal Training.

Wraps GNS_heterogeneous to extract physics-informed embeddings
and aggregate them for temporal modeling (TGT).
"""

import torch
import torch.nn as nn
from torch_scatter import scatter_add
from typing import Dict, Tuple

from gridfm_graphkit.models.gnn_heterogeneous_gns import GNS_heterogeneous


class GridFMEncoder(nn.Module):
    """
    Wrapper around GNS_heterogeneous for spatio-temporal forecasting.
    
    Responsibilities:
        1. Extract physics-informed embeddings (h_bus, h_gen)
        2. Aggregate generator embeddings onto parent buses (Option C)
        3. Optionally return reconstruction outputs for multi-task training
    
    Args:
        gridfm_model: An instantiated GNS_heterogeneous model
        
    Attributes:
        output_dim: Embedding dimension (hidden_dim * heads)
        
    Example:
        >>> encoder = GridFMEncoder(gridfm_model)
        >>> encoder.reset_parameters()  # Fresh init for end-to-end training
        >>> out = encoder(x_dict, edge_index_dict, edge_attr_dict, mask_dict)
        >>> embeddings = out["embedding"]  # [N_bus, hidden_dim * heads]
    """
    
    def __init__(self, gridfm_model: GNS_heterogeneous):
        super().__init__()
        self.gridfm = gridfm_model
        
        # Extract architecture info for external access
        self.hidden_dim = self.gridfm.hidden_dim
        self.heads = self.gridfm.heads
        self.output_dim = self.hidden_dim * self.heads
        self.num_layers = self.gridfm.num_layers
        self.task = self.gridfm.task
    
    def reset_parameters(self) -> None:
        """
        Reset all parameters for fresh end-to-end training.
        
        Uses Xavier uniform for Linear layers, ones/zeros for LayerNorm,
        and calls native reset_parameters() where available.
        """
        reset_count = 0
        for name, module in self.gridfm.named_modules():
            if hasattr(module, 'reset_parameters'):
                try:
                    module.reset_parameters()
                    reset_count += 1
                except TypeError:
                    pass
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                reset_count += 1
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
                reset_count += 1
        
        print(f"GridFMEncoder: Reset {reset_count} modules")
    
    def _aggregate_gen_to_bus(
        self,
        h_bus: torch.Tensor,
        h_gen: torch.Tensor,
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> torch.Tensor:
        """
        Aggregate generator embeddings onto their parent buses.
        
        This implements "Option C" from our discussion:
        - Preserves IEEE-14 topology (14 nodes, not 19)
        - Generator info is merged into the bus it's connected to
        
        Args:
            h_bus: Bus embeddings [N_bus, D]
            h_gen: Generator embeddings [N_gen, D]
            edge_index_dict: Contains ("gen", "connected_to", "bus") edge index
            
        Returns:
            h_merged: Enriched bus embeddings [N_bus, D]
        """
        num_bus = h_bus.size(0)
        
        # edge_index[1] = target bus index for each generator
        gen_to_bus_index = edge_index_dict[("gen", "connected_to", "bus")][1]
        
        # Sum generator embeddings at their connected buses
        h_gen_aggregated = scatter_add(
            src=h_gen,
            index=gen_to_bus_index,
            dim=0,
            dim_size=num_bus,
        )
        
        # Combine: bus + aggregated generator info
        return h_bus + h_gen_aggregated
    
    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_attr_dict: Dict[Tuple[str, str, str], torch.Tensor],
        mask_dict: Dict[str, torch.Tensor],
        return_reconstruction: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional multi-task outputs.
        
        Args:
            x_dict: Node features {"bus": [N_bus, F], "gen": [N_gen, F]}
            edge_index_dict: Edge connectivity for each relation type
            edge_attr_dict: Edge attributes (G, B for branches)
            mask_dict: Masks indicating which values to predict
            return_reconstruction: If True, include physics predictions
            
        Returns:
            Dictionary containing:
                "embedding": Merged bus embeddings [N_bus, D]
                             D = hidden_dim * heads (e.g., 48 * 8 = 384)
                             
            If return_reconstruction=True, also includes:
                "pred_bus": Physics-decoded predictions [N_bus, output_bus_dim]
                "pred_gen": Generator predictions [N_gen, output_gen_dim]
                "h_bus": Raw bus embeddings [N_bus, D]
                "h_gen": Raw generator embeddings [N_gen, D]
        """
        # Call GridFM with return_embeddings=True
        gridfm_out = self.gridfm(
            x_dict, 
            edge_index_dict, 
            edge_attr_dict, 
            mask_dict,
            return_embeddings=True,
        )
        
        # Extract embeddings
        h_bus = gridfm_out["h_bus"]  # [N_bus, hidden_dim * heads]
        h_gen = gridfm_out["h_gen"]  # [N_gen, hidden_dim * heads]
        
        # Aggregate generators onto buses (Option C)
        h_merged = self._aggregate_gen_to_bus(h_bus, h_gen, edge_index_dict)
        
        # Build output
        output = {"embedding": h_merged}
        
        if return_reconstruction:
            output["pred_bus"] = gridfm_out["bus"]
            output["pred_gen"] = gridfm_out["gen"]
            output["h_bus"] = h_bus
            output["h_gen"] = h_gen
        
        return output
    
    def get_output_dim(self) -> int:
        """Return embedding dimension for downstream models."""
        return self.output_dim
    
    def get_layer_residuals(self) -> Dict[int, torch.Tensor]:
        """Return physics residuals from last forward pass (for monitoring)."""
        return self.gridfm.layer_residuals
    
    def __repr__(self) -> str:
        trainable = any(p.requires_grad for p in self.gridfm.parameters())
        return (
            f"GridFMEncoder(\n"
            f"  task={self.task},\n"
            f"  hidden_dim={self.hidden_dim},\n"
            f"  heads={self.heads},\n"
            f"  output_dim={self.output_dim},\n"
            f"  num_layers={self.num_layers},\n"
            f"  trainable={trainable}\n"
            f")"
        )