import json
import pandas as pd

from fastapi import Response, Request

from google.cloud.aiplatform.prediction.handler import PredictionHandler

class CprHandler(PredictionHandler):
    async def handle(self, request: Request):
        try:
            request_body = await request.body()
            print('Prediction data received')
            
            request_data = json.loads(request_body.decode('utf-8'))
            print(type(request_data['instances'][0]))
            try:
                if request_data['request_type'] == 'Online':
                    is_online_prediction = True
                else:
                    is_online_prediction = False
            except:
                is_online_prediction = False
            
            print('Online Prediction Detected' if is_online_prediction else 'Batch Prediction Detected')
                
            prediction_instances = pd.DataFrame(request_data['instances'][0])
            print(prediction_instances.head())
            print('DataFrame created from request data')

            df = self._predictor.preprocess(data = prediction_instances)
            print('Preprocessing completed')
            
            self._predictor.predict()
            print('Prediction completed')
            
            prediction_results = self._predictor.post_process(data = df)
            print('Postprocessing completed')
            
            respone_json = json.dumps(prediction_results)
            
            if is_online_prediction:
                return Response(content = respone_json, media_type = 'application/json')
            else:
                json_output = json.dumps({'predictions':[prediction_results]})
                return Response(content = json_output)
        
        except Exception as e:
            message = 'Exception : ' + str(e)
            return Response(content = json.dumps({
                'code' : 500,
                'Message' : message
            }))