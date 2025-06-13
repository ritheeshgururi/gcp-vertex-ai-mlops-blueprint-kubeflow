import numpy as np
import pandas as pd

from google.cloud.aiplatform.utils import prediction_utils
from abc import ABC

from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer

class CprPredictor(ABC):
    def __init__(self):
        return
    
    def load(self, artifacts_uri: str):
        prediction_utils.download_model_artifacts(artifacts_uri)
        self._best_tft = TemporalFusionTransformer.load_from_checkpoint('tft_model_ckpt.ckpt')
    
    def preprocess(self, data:pd.DataFrame):
        try:
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

            print('Data received and read here in preprocess')
            data['date'] = pd.to_datetime(data['date'])

            #add time index
            print('convert date to datetime')
            data['time_idx'] = data['date'].dt.year * 12 + data['date'].dt.month
            data['time_idx'] -= data['time_idx'].min()

            #add features
            print('created time index')
            data['month'] = data['date'].dt.month.astype(str).astype('category')
            data['log_volume'] = np.log(data.volume + 1e-8)
            data['avg_volume_by_sku'] = data.groupby(['time_idx', 'sku'], observed=True).volume.transform('mean')
            data['avg_volume_by_agency'] = data.groupby(['time_idx', 'agency'], observed=True).volume.transform('mean')
            print('add additioinal features')

            data[special_days] = data[special_days].apply(lambda x: x.map({0: '-', 1: x.name})).astype('category')
            max_encoder_length = 24
            print('convert special days to category')
            print(data.info())

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
            print('create time series dataset')

            self._batch_dataloader = batch_dataset.to_dataloader(train = False, batch_size = 128, num_workers = 3)
            print('create dataloader')
        except Exception as e:
            print('PreProcess Exception: ', e)
        return data
    
    def predict(self):
        try:
            self._raw_predictions = self._best_tft.predict(self._batch_dataloader, mode = 'raw', return_index = True, return_x = True)
        except Exception as e:
            print('Predict Exception: ', e)
        
    def post_process(self, data:pd.DataFrame):
        try:
            predictions_df = data.copy()
            predictions_df['predicted_volume'] = np.nan
            predictions_df['date'] = predictions_df['date'].astype(str)
            print('convert date to string')

            median_predictions = self._raw_predictions.output.prediction.cpu().numpy()[:,:,4] 
            print('get median predictions')

            for i, row in self._raw_predictions.index.iterrows():
                predictions_df.loc[(predictions_df['agency'] == row['agency']) & (predictions_df['sku'] == row['sku']), 'predicted_volume'] = median_predictions[i][0]
            print('create final df with predictions')
        except Exception as e:
            print('PostProcess Exception: ', e)
        
        return predictions_df.to_dict(orient='records')