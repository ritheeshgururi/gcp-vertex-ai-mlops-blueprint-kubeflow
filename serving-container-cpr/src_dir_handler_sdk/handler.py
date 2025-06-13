import json
import pandas as pd

from fastapi import Response, Request

from google.cloud.aiplatform.prediction.handler import PredictionHandler

class CprHandler(PredictionHandler):
    async def handle(self, request: Request):
        try:
            request_body = await request.body()
            print('Data Received')
            
            request_data = json.loads(request_body.decode('utf-8'))
            print(type(request_data['instances'][0]))
            try:
                if request_data['request_type'] == 'Online':
                    is_online_prediction = True
                    data_list = json.loads(request_data['instances'][0])
                else:
                    is_online_prediction = False
                    data_list = request_data['instances'][0]
            except:
                is_online_prediction = False
                data_list = request_data['instances'][0]
            
            print('Online Prediction Detected' if is_online_prediction else 'Batch Prediction Detected')
                
            prediction_instances = pd.DataFrame(request_data['instances'][0])
            print(prediction_instances.head())
            print('DataFrame Created')

            df = self._predictor.preprocess(data = prediction_instances)
            print('Preprocess Done')
            
            self._predictor.predict()
            print('Prediction Done')
            
            prediction_results = self._predictor.post_process(data = df)
            print('Postprocess Done')
            
            json_res = json.dumps(prediction_results)
            
            if is_online_prediction:
                return Response(content = json_res, media_type = 'application/json')
            else:
                json_output = json.dumps({'predictions':[prediction_results]})
                return Response(content=json_output)
        
        except Exception as e:
            message = 'Exception : ' + str(e)
            return Response(content = json.dumps({
                'code' : 500,
                'Message' : message
            }))