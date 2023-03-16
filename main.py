import os
import io
import PIL
import json
import uvicorn
import requests

import numpy as np
from io import BytesIO
from PIL import Image, ImageOps
from fastapi import FastAPI

from typing import Optional
from pydantic import BaseModel
from nsfw_detector import predict
from keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="Image Classification",
    description="classify images into different categories")


# origins = [
#     "https://ai.propvr.tech",
#     "http://ai.propvr.tech",
#     "https://getedge.glitch.me",
#     "http://getedge.glitch.me",
#     "https://getedge.glitch.me/*",
#     "http://getedge.glitch.me/*",
#     "https://app.smartagent.ae",
#     "https://localhost:8081",
#     "https://uatapp.smartagent.ae",
#     "http://app.smartagent.ae",
#     "http://localhost:8081",
#     "http://uatapp.smartagent.ae",
#     "https://app.smartagent.ae/*",
#     "https://localhost:8081/*",
#     "https://uatapp.smartagent.ae/*",
#     "http://app.smartagent.ae/*",
#     "http://localhost:8081/*",
#     "http://uatapp.smartagent.ae/*"
#     ]


# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# Load the model for irrelavent images(+18)
model_nsfw = predict.load_model('./nsfw_mobilenet2.224x224.h5')

# Load the model for image Label(Classification)
model = load_model('keras_model_tm1.h5',compile=False)

#-------------------------------++++++++--------------------------------------------------

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model for checking images is valid or not
model_path = "keras_model_new2.h5"
m_path = os.path.basename(model_path)
model_other = load_model(m_path, compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

#--------------------------------+++++++--------------------------------------

@app.get("/")
async def root():
    return "Server is up!"
   

def predict_img(image: Image.Image):

    data = np.ndarray(shape=(1,224, 224, 3), dtype=np.float32)
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    img_array_final = normalized_image_array[:, :, :3]
    data[0] = img_array_final

    # Predicts the image is belong to real eastate image or not
    prediction = model_other.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    
    if class_name[2:] == "Not_valid\n":
    
        prediction_dict= {
        "response": {
            "solutions": {
                "re_roomtype_eu_v2": {
                     
                     "predictions": [
                                        {
                                        "confidence": str(confidence_score),
                                        "label" : "False"
                                        }
                                    ],
                                    "top_prediction":{
                                        "confidence": str(confidence_score),
                                            "label": "False"
                                    }
                                }
                            }
                        }
                    }
        
        return prediction_dict
    else:
        #labels = ['Bathroom','Room-bedroom','Living_Room','Outdoor_building','Kitchen','Non_Related','Garden','Plot','Empty_room']
        labels = ['balcony','bathroom','bedroom','corridor','dining_room','exterior_view','gym','kitchen','lift','living_room','parking','stairs','swimming_pool','utility_room','others']


        data = np.ndarray(shape=(1,224, 224, 3), dtype=np.float32)
        size = (224, 224)

        image = ImageOps.fit(image, size, Image.ANTIALIAS)
        image_array = np.asarray(image)
        #image.show()
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        img_array_final = normalized_image_array[:, :, :3]
        #print("shape: ", img_array_final.shape)
        data[0] = img_array_final
        result = model.predict(data)
        #result1 = sorted(result, reverse=True)
        
        arr_sorted = -np.sort(-result,axis=1)
        top_five = arr_sorted[:,:5]
        top_five_array = result.argsort()
        top_five_array1 = top_five_array.tolist()
        top1 = top_five_array1[0][-1]
        top2 = top_five_array1[0][-2]
        top3 = top_five_array1[0][-3]
        top4 = top_five_array1[0][-4]
        top5 = top_five_array1[0][-5]

        index_max = np.argmax(result)


        
        prediction_dict = {
            "response": {
                "solutions": {
                    "re_roomtype_eu_v2": {
                        
                        "predictions": [
                            {
                                "confidence": str(top_five[0][0]),
                                "label": str(labels[top1])
                            },
                            {
                                "confidence": str(top_five[0][1]),
                                "label": str(labels[top2])
                            },
                            {
                                "confidence": str(top_five[0][2]),
                                "label": str(labels[top3])
                            },
                            {
                                "confidence": str(top_five[0][3]),
                                "label": str(labels[top4])
                            },
                            {
                                "confidence": str(top_five[0][4]),
                                "label": str(labels[top5])
                            }
                        ],

                       
                        "top_prediction":{
                            "confidence": str(result[0][index_max]),
                                "label": str(labels[index_max])
                        }
                    }
                }
            }
        }

        if labels[top1] == 'others' and top_five[0][0]*100 > 80:

            prediction_dict= {
                    "response": {
                        "solutions": {
                            "re_roomtype_eu_v2": {
                                
                                "predictions": [
                                                    {
                                                    "confidence": str(top_five[0][0]),
                                                    "label" : "False"
                                                    }
                                                ],
                                                "top_prediction":{
                                                    "confidence": str(top_five[0][0]),
                                                        "label": "False"
                                                }
                                            }
                                        }
                                    }
                                }
            return prediction_dict
        
        else:

            return prediction_dict


@app.get("/predict_from_url")
async def predict_image(image_url: str):

    response = requests.get(image_url)
    image_bytes = io.BytesIO(response.content)
    input_image = Image.open(image_bytes).convert("RGB")
    input_image.save("input_img.jpg")
    
    result = predict.classify(model_nsfw,"input_img.jpg")
    status_check = result["input_img.jpg"]

    final_result = status_check['porn']*100
    final_result2 = status_check['sexy']*100

    if  final_result > 30 or final_result2 > 10 :

        prediction_dict= {
        "response": {
            "solutions": {
                "re_roomtype_eu_v2": {
                     
                     "predictions": [
                                        {
                                        "confidence": str(final_result),
                                        "label" : "False"
                                        }
                                    ],
                                    "top_prediction":{
                                        "confidence": str(final_result),
                                            "label": "False"
                                    }
                                }
                            }
                        }
                    }
        
        return prediction_dict
    
    else:

        function = predict_image1(image_url)
        return function

def predict_image1(str_url: str):

    response = requests.get(str_url)
    image_bytes = io.BytesIO(response.content)
    img = Image.open(image_bytes)
    prediction1 = predict_img(img)
    data = json.dumps(prediction1)
    data1 = json.loads(data.replace("\'", '"'))

    return data1

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
