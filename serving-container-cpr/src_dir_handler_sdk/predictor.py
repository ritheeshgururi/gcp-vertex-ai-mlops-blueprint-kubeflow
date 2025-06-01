import logging
import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from google.cloud.aiplatform.utils import prediction_utils
from abc import ABC

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CprPredictor(ABC):
    def load(self, artifacts_uri: str):
        prediction_utils.download_model_artifacts(artifacts_uri)
        self._best_tft = TemporalFusionTransformer.load_from_checkpoint('tft_model.ckpt')
    
    def preprocess(self, data:pd.DataFrame):
        try:
            logger.info('Prediction data loaded')
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

            logger.info('Preprocessing started')
            data['date'] = pd.to_datetime(data['date'])

            #add time index
            logger.info('Adding time index')
            data['time_idx'] = data['date'].dt.year * 12 + data['date'].dt.month
            data['time_idx'] -= data['time_idx'].min()

            #add features
            logger.info('Adding features')
            data['month'] = data['date'].dt.month.astype(str).astype('category')
            data['log_volume'] = np.log(data.volume + 1e-8)
            data['avg_volume_by_sku'] = data.groupby(['time_idx', 'sku'], observed=True).volume.transform('mean')
            data['avg_volume_by_agency'] = data.groupby(['time_idx', 'agency'], observed=True).volume.transform('mean')

            data[special_days] = data[special_days].apply(lambda x: x.map({0: '-', 1: x.name})).astype('category')
            max_encoder_length = 24
            logger.info(data.info())

            logger.info('Starting TimeSeriesDataSet object creation')
            batch_dataset = TimeSeriesDataSet(
                data,
                time_idx = 'time_idx',
                target = 'volume',
                group_ids = ['agency', 'sku'],
                min_encoder_length = max_encoder_length // 2,
                max_encoder_length = max_encoder_length,
                min_prediction_length = 1,
                max_prediction_length = 1,
                static_categoricals = ['agency', 'sku'],
                static_reals = ['avg_population_2017', 'avg_yearly_household_income_2017'],
                time_varying_known_categoricals = ['special_days', 'month'],
                variable_groups = {'special_days': special_days},
                time_varying_known_reals = ['time_idx', 'price_regular', 'discount_in_percent'],
                time_varying_unknown_categoricals = [],
                time_varying_unknown_reals = [
                    'volume',
                    'log_volume',
                    'industry_volume',
                    'soda_volume',
                    'avg_max_temp',
                    'avg_volume_by_agency',
                    'avg_volume_by_sku',
                ],
                target_normalizer = GroupNormalizer(
                    groups = ['agency', 'sku'], transformation = 'softplus'
                ),
                add_relative_time_idx = True,
                add_target_scales = True,
                add_encoder_length = True
            )
            logger.info('TimeSeriesDataSet object creation completed')

            logger.info('Starting dataloader creation')
            self._batch_dataloader = batch_dataset.to_dataloader(train = False, batch_size = 128, num_workers = 3)
            logger.info('Dataloader creation complete')
        except Exception as e:
            logger.info('Error during preprocessing: ', e)
        return data
    
    def predict(self):
        try:
            logger.info('Starting prediction')
            self._raw_predictions = self._best_tft.predict(self._batch_dataloader, mode = 'raw', return_index = True, return_x = True)
            logger.info('Prediction completed')
        except Exception as e:
            logger.info('Error during prediction: ', e)
        
    def post_process(self, data:pd.DataFrame):
        try:
            predictions_df = data.copy()
            predictions_df['predicted_volume'] = np.nan
            predictions_df['date'] = predictions_df['date'].astype(str)
            logger.info('Converted date to string')

            median_predictions = self._raw_predictions.output.prediction.cpu().numpy()[:,:,4] 
            logger.info('Obtained median predictions')

            for i, row in self._raw_predictions.index.iterrows():
                predictions_df.loc[(predictions_df['agency'] == row['agency']) & (predictions_df['sku'] == row['sku']), 'predicted_volume'] = median_predictions[i][0]
            logger.info('Final dataframe with predictions created')
        except Exception as e:
            logger.info('Error during Postprocessing: ', e)
        
        return predictions_df.to_dict(orient = 'records')