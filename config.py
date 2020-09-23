

class args:
    
    exp_name = "base_model"
    sub_name = ""
    output_dir = "weights"

    network = "GRU_model"

    losses  = "MCRMSELoss"

    
    # Model parameters
    num_embeddings = 14
    embedding_dim  = 128
    hidden_layers  = 3
    hidden_size    = 128
    dropout        = 0.5

    # Training parameters
    lr = 0.0001
    seed = 42
    epochs = 50
    n_folds = 5
    batch_size = 32
