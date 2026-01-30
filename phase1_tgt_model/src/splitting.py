import numpy as np

def get_interleaved_splits(T, config):
    """
    Implements repeated 4:1:1 Splitting.
        # 0,1,2,3 -> Train
        # 4       -> Val
        # 5       -> Test    
    Returns:
        train_idx, val_idx, test_idx (arrays of time steps)
    """
    n_blocks = T // config["BLOCK_SIZE_HOURS"]
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    for i in range(n_blocks):
        start_t = i * config["BLOCK_SIZE_HOURS"]
        end_t = start_t + config["BLOCK_SIZE_HOURS"]
        cycle_pos = i % config["CYCLE_SCHEME"]
        block_range = np.arange(start_t, end_t)
        
        if cycle_pos in [0, 1, 2, 3]:
            train_indices.extend(block_range)
            
        elif cycle_pos == 4:
            if start_t >= config["INPUT_WINDOW"]:
                val_indices.extend(block_range)
            else: 
                train_indices.extend(block_range) # Fallback
                
        elif cycle_pos == 5:
            if start_t >= config["INPUT_WINDOW"]:
                test_indices.extend(block_range)
            else:
                train_indices.extend(block_range)
    return np.array(train_indices), np.array(val_indices), np.array(test_indices)


def get_temporal_splits(T):
    """
    Strict temporal (chronological) split for forecasting.
    Train: 0% - 70%
    Val:   70% - 85%
    Test:  85% - 100%
    """
    split_train = int(0.70 * T)
    split_val = int(0.85 * T)
    
    train_idx = np.arange(0, split_train)
    val_idx = np.arange(split_train, split_val)
    test_idx = np.arange(split_val, T)
    
    return train_idx, val_idx, test_idx