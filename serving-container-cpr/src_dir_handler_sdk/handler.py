import json
import logging
from fastapi import Response, Request
import pandas as pd
from google.cloud.aiplatform.prediction.handler import PredictionHandler

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

class CprHandler(PredictionHandler):
    async def handle(self, request: Request):
        try:
            request_body = await request.body()
            logger.info("Request received")
            
            request_data = json.loads(request_body.decode("utf-8"))
            logger.info(type(request_data["instances"][0]))
            try:
                if request_data['request_type'] == "Online":
                    is_online_prediction = True
                else:
                    is_online_prediction = False
            except:
                is_online_prediction = False
            
            logger.info("Online Prediction Detected" if is_online_prediction else "Batch Prediction Detected")
                
            prediction_instances = pd.DataFrame(request_data["instances"][0])
            print(prediction_instances.head())
            logger.info("Prediction dataFrame created")

            df = self._predictor.preprocess(data = prediction_instances)
            logger.info("Exited preprocessing")
            
            self._predictor.predict()
            logger.info("Exited prediction")
            
            prediction_results = self._predictor.post_process(data = df)
            logger.info("Exited prediction")
            
            json_res = json.dumps(prediction_results)
            
            if is_online_prediction:
                return Response(content = json_res, media_type = "application/json")
            else:
                json_output = json.dumps({"predictions":[prediction_results]})
                return Response(content = json_output)
        except Exception as e:
            message = "Exception : " + str(e)
            return Response(content = json.dumps({
                "code" : 500,
                "Message" : message
            }))