## Tesis Repo description
1. phase1_baseline folder is an experiment to forecast loads only. It is finished for now. 
2. phase2_ST-GNN folder is an experiment to build a spatio temporal graph transformer to forecast power flows. This is work in progress. 
    - i copied gnn_heterogeneous_gns from graphkit to thesis_repo and modified the model slightly because i need more outputs. This is named `gnn_heterogeneous_gns.py`.
        -> Modification: If `return_embeddings=True`, the model returns the hidden intermediate tensors `h_bus` and `h_gen` (immediately before the final MLP projection heads).
    
    - The Wrapper.py extracts the hidden embeddings of gnn_heterogeneous_gns_thesis right before the gnn_heterogeneous_gns_thesis.

### Next steps:
    - Abstract: The plan is next, to adapt the model Tiny TGT model from phase1_baseline\src\models.py to become a spatio temporal graph transformer to forecast power flows. 

    - PLAN:
        - TinyTGT Adaption: the TinyTGT has to be modified (create new script):
            1.  new TGT takes embeddings from Wrapper as spatial input
                - `[B, T, N_bus, D]`
            2. embeddings are input in temporal layer (self attention)
            3. NN makes prediction of power flow vars for forecast horizon
            4. The foercasts have to be checked if physical valid (Power flow equations satisfied). For this two ideas:
                a) initial: If prediction infeasible -> put into newton raphson solver
                b) later/more complex: Use Feasebility Restauration Layer like the TU_DElft guys
        - Training:
            - Loss function = Forecasting error + GNN error:
                - Forecasting error: 
                    a) calc power flow residuals and give as training signal
                    b) calc deviation from true power flow vals
                    c) GNN error: stays as is (physics informed loss)

### Open Questions: 
1. Temporal Encodings:
    - Start without?
    - Where/How to add temporal features (time of day, day of week) to input features? 
    - Learnable Temporal Embeddings vector before Attention insertable?
2. Start with Newton Rapson to correct infeasable output, or FRL (Feasebility Restauration Layer like the TU_DElft )?
3. Start with multi-step ahead prediction (24 hours) later compare vs autoregressive approach?