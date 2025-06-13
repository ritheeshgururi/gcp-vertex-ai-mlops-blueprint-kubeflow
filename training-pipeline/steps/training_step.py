import logging

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

from configuration.step_config import TrainingConfig
from utils import step_utils

def training_step(
    project,
    location,
    artifact_bucket,
    vertex_experiment_name,
    vertex_run_name,
    training_input,
    train_loader_input,
    val_loader_input,
    best_params_input,
    pth_model_output,
    ckpt_model_output
):
    """Trains the TFT time series forecasting model"""

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
        
    ###fetching model training variables/parameters from the step_config.py file
    max_epochs = TrainingConfig.max_epochs
    early_stopping_monitor = TrainingConfig.early_stopping_monitor
    early_stopping_min_delta = TrainingConfig.early_stopping_min_delta
    early_stopping_patience = TrainingConfig.early_stopping_patience
    early_stopping_mode = TrainingConfig.early_stopping_mode
    trainer_accelerator = TrainingConfig.trainer_accelerator
    trainer_gradient_clip_val = TrainingConfig.trainer_gradient_clip_val
    trainer_limit_train_batches = TrainingConfig.trainer_limit_train_batches
    tft_log_interval = TrainingConfig.tft_log_interval
    tft_optimizer = TrainingConfig.tft_optimizer
    tft_reduce_on_plateau_patience = TrainingConfig.tft_reduce_on_plateau_patience
    
    ###logging parameters to the vertex experiment run
    logger.info('Initializing Vertex AI with experiment name and instantiating experiment run')
    run = step_utils.get_vertex_experiment_run(
        project,
        location,
        vertex_experiment_name,
        vertex_run_name
    )

    logger.info('Logging parameters to the experiment run')
    run.log_params({
        'TRAINING_CONFIG_max_epochs': max_epochs,
        'TRAINING_CONFIG_early_stopping_monitor': early_stopping_monitor,
        'TRAINING_CONFIG_early_stopping_min_delta': early_stopping_min_delta,
        'TRAINING_CONFIG_early_stopping_patience': early_stopping_patience,
        'TRAINING_CONFIG_early_stopping_mode': early_stopping_mode,
        'TRAINING_CONFIG_trainer_accelerator': trainer_accelerator,
        'TRAINING_CONFIG_trainer_gradient_clip_val': trainer_gradient_clip_val,
        'TRAINING_CONFIG_trainer_limit_train_batches': trainer_limit_train_batches,
        'TRAINING_CONFIG_tft_log_interval': tft_log_interval,
        'TRAINING_CONFIG_tft_optimizer': tft_optimizer,
        'TRAINING_CONFIG_tft_reduce_on_plateau_patience': tft_reduce_on_plateau_patience
    })
    
    ###start of model training logic
    logger.info('Starting model training')
    
    # Load raw data
    training = step_utils.load_data_from_component_input_in_pickle(training_input)
    train_dataloader = step_utils.load_data_from_component_input_in_pickle(train_loader_input)
    val_dataloader = step_utils.load_data_from_component_input_in_pickle(val_loader_input)
    best_params = step_utils.load_data_from_component_input_in_pickle(best_params_input)
    
    early_stop_callback = EarlyStopping(
        monitor = early_stopping_monitor,
        min_delta = early_stopping_min_delta,
        patience = early_stopping_patience,
        verbose = False,
        mode = early_stopping_mode
    )
    lr_logger = LearningRateMonitor()
    tb_logger = TensorBoardLogger('lightning_logs')
    
    trainer = pl.Trainer(
        max_epochs = max_epochs,
        accelerator = trainer_accelerator,
        enable_model_summary = True,
        gradient_clip_val = trainer_gradient_clip_val,
        limit_train_batches = trainer_limit_train_batches,
        callbacks = [lr_logger, early_stop_callback],
        logger = tb_logger
    )
    
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate = best_params['learning_rate'],
        hidden_size = best_params['hidden_size'],
        attention_head_size = best_params['attention_head_size'],
        dropout = best_params['dropout'],
        hidden_continuous_size = best_params['hidden_continuous_size'],
        loss = QuantileLoss(),
        log_interval = tft_log_interval,
        optimizer = tft_optimizer,
        reduce_on_plateau_patience = tft_reduce_on_plateau_patience
    )
    
    trainer.fit(
        tft,
        train_dataloaders = train_dataloader,
        val_dataloaders = val_dataloader
    )
    
    metrics = trainer.callback_metrics
    
    run.log_metrics({
        'val_loss': float(metrics['val_loss'].cpu()),
        'train_loss': float(metrics['train_loss'].cpu())
    })
    
    #save the trained model as PyTorch .pth
    model_dict = {
        'state_dict': tft.state_dict(),
        'hparams': tft.hparams,
        'training_config': training.get_parameters()
    }
    torch.save(model_dict, pth_model_output)
    
    #save the trained model as checkpoint ckpt
    trainer.save_checkpoint(ckpt_model_output)
    ###end of model training logic
    
    #Saving the model pth and ckpt file to GCS
    model_pth_path = f'{vertex_run_name}/training_artifacts/tft_model_artifact/tft_model_pth.pth'
    model_ckpt_path = f'{vertex_run_name}/training_artifacts/ckpt_model_artifact/tft_model_ckpt.ckpt'

    logger.info(f'Archiving preprocessed data to GCS path: gs://{artifact_bucket}/{vertex_run_name}/training_artifacts/')
    step_utils.upload_file_to_gcs(
        project,
        artifact_bucket,
        pth_model_output,
        model_pth_path
    )
    step_utils.upload_file_to_gcs(
        project,
        artifact_bucket,
        ckpt_model_output,
        model_ckpt_path
    )
    logger.info(f'Model trained and archived to GCS path: gs://{artifact_bucket}/{vertex_run_name}/training_artifacts/')
    
    run.log_params({
        'TRAINING_OUTPUT_pth_model_artifact_uri': f'gs://{artifact_bucket}/{model_pth_path}',
        'TRAINING_OUTPUT_ckpt_model_artifact_uri': f'gs://{artifact_bucket}/{model_ckpt_path}'
    })
    logger.info('Model artifact URI logged as a parameter to experiment run')