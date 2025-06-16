class PreprocessConfig:
    special_days = [
        'easter_day',
        'good_friday',
        'new_year',
        'christmas',
        'labor_day',
        'independence_day',
        'revolution_day_memorial',
        'regional_games',
        'fifa_u_17_world_cup',
        'football_gold_cup',
        'beer_capital',
        'music_fest'
    ]
    volume_transform_type = 'mean'
    
class DataloaderConfig:
    batch_size = 128
    max_prediction_length = 6
    max_encoder_length = 24  
    min_max_encoder_length_ratio = 2
    min_prediction_length = 1
    target_normalizer_transformation = 'softplus'
    num_workers = 0
    train_val_batch_size_ratio = 10
    
class HPTConfig:
    n_trials = 5
    max_epochs = 5
    gradient_clip_val_range = (0.01, 1.0)
    hidden_size_range = (8, 128)
    hidden_continuous_size_range = (8, 128)
    attention_head_size_range = (1, 4)
    learning_rate_range = (0.001, 0.1)
    dropout_range = (0.1, 0.3)
    trainer_kwargs_limit_train_batches = 30
    reduce_on_plateau_patience = 4
    use_learning_rate_finder = False
    
class TrainingConfig:
    max_epochs = 5
    early_stopping_monitor = 'val_loss'
    early_stopping_min_delta = 1e-4
    early_stopping_patience = 10
    early_stopping_mode = 'min'
    trainer_accelerator = 'cpu'
    trainer_gradient_clip_val = 0.1
    trainer_limit_train_batches = 50
    tft_log_interval = 10
    tft_optimizer = 'ranger'
    tft_reduce_on_plateau_patience = 4
    
class DeployConfig:
    deploy_as_version = True
    if deploy_as_version:
        PARENT_MODEL = '8590537124372021248' 
    SERVING_CONTAINER_IMAGE_URI = 'asia-south1-docker.pkg.dev/gcp-vertexai-mlops-blueprint/tft-cpr-serving-container/serving-container-cpr:latest'
    MODEL_DISPLAY_NAME = 'TFT CPR Model'
    SERVING_CONTAINER_PREDICT_ROUTE = '/predict'
    SERVING_CONTAINER_HEALTH_ROUTE = '/health'
    
    ENDPOINT_DISPLAY_NAME = 'TFT Model Endpoint'
    SERVING_CONTAINER_MACHINE_TYPE = 'n2-standard-2'
    deploy_to_existing_endpoint = True
    if deploy_to_existing_endpoint:
        EXISTING_ENDPOINT_ID = '5209024297519022080'