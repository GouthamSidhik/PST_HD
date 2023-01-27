import time, tensorflow as tf, os, cv2, pandas as pd
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import numpy as np
from PIL import Image
from pytesseract import pytesseract
import re
import matplotlib.pyplot as plt
import base64, io, pymssql
import pyinputplus as pyip
import warnings
warnings.filterwarnings('ignore') 
'''
#2k trained model
PATH_TO_CFG="E:/PROJECTS/P_1/Tensorflow/workspace/train_model/exported_models/my_model/pipeline.config"
PATH_TO_CKPT="E:/PROJECTS/P_1/Tensorflow/workspace/train_model/exported_models/my_model/checkpoint"
'''
PATH_TO_CFG=input('path to config: ') #model/pipeline.config
PATH_TO_CKPT=input('path to checkpoint: ') #model/checkpoint


print('Loading model... ', end='')
start_time = time.time()

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()

@tf.function
def detect_fn(image):

    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

#PATH_TO_LABELS = 'E:/PROJECTS/P_1/Tensorflow/workspace/train_model/annotations/label_map.pbtxt'

PATH_TO_LABELS=input('path to labels: ') #label_map.pbtxt file

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,use_display_name=True)

#pytesseract.tesseract_cmd='C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

pytesseract.tesseract_cmd=input('path to tesseractOCR: ') # tesseract.exe file
    
q=[]#height
p=[]#rollno
p1=[]#error
s=[]#detected_height


server=input('server: ')
user=input('user: ')
password = pyip.inputPassword(prompt='password: ', mask='*')
database=input('database: ')

db = pymssql.connect(server,user,password,database)
cursor = db.cursor()
print('connection established')

sql = ("select Results.RollNo,imagedata, Height from Results,ImageDataPMT where Results.Rollno=ImageDataPMT.rollno ")

cursor.execute(sql)
data = cursor.fetchall()
db.close()
print('Data retreived and connection closed')

print('Processing...')

start_time_1 = time.time()

for i in range(len(data)):

    image_base64=data[i][1]
    rollno=data[i][0]
    act_height=data[i][2]  
    
    if image_base64 == "":
        p1.append(rollno)
        continue
    base64_decoded = base64.b64decode(image_base64)
    
    image = Image.open(io.BytesIO(base64_decoded))

    image_np = np.array(image)


    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()
    ogimg=image_np
    viz_utils.visualize_boxes_and_labels_on_image_array(
          ogimg,
          detections['detection_boxes'],
          detections['detection_classes']+label_id_offset,
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=1,
          min_score_thresh=.30,
          agnostic_mode=False)  
    w,h=640,480 
    oimg=image
    for y in range(len(detections['detection_boxes'])):
        if (detections['detection_scores'][y]==detections['detection_scores'].max()):
            a=detections['detection_boxes'][y]
            y1=a[0]
            x1=a[1]
            y2=a[2]
            x2=a[3]
            
            (l,t,r,b)=(x1*w,y1*h,x2*w,y2*h)
            box=(l,t,r,b)
            im=oimg.crop(box)
            #im.show() #to show the cropped image


    text=pytesseract.image_to_string(im,lang='eng',config='--psm 6,--oem 2')

    regex = re.compile(r'\d{3}\.\d{1}|\b\d{3}\b')
    t = regex.findall(text)

    t=[float(z) for z in t]

    if (len(t)<1):
        p1.append(rollno)
        #print (i,'>>> Image Error can not detect numbers')
        #print('----------') 

    else:
        if (len(t)>1):
            pred_height=sum(t)/len(t)
            if (pred_height<250):
                #print(i,'>>>',pred_height)
                if abs(act_height-pred_height)>0.5:
                    p1.append(rollno)
                else:   
                    p.append(rollno)
                    q.append(act_height)
                    s.append(pred_height)
                
            else:
                #print (i,'>>> Number Detection Error')
                p1.append(rollno)

        else:
            if (t[0]<250):
                pred_height=t[0]
                #print(i,'>>>',pred_height)
                if abs(act_height-pred_height)>0.5:
                    p1.append(rollno)
                else:
                    p.append(rollno)
                    q.append(act_height)
                    s.append(pred_height)
                
            else:
                #print (i,'>>> Number Detection Error')
                p1.append(rollno)
                
        #print('----------')              

        cv2.waitKey(0) # waits until a key is pressed
        cv2.destroyAllWindows()

    if i % 100 == 0:
        o=len(data)-i
        print(i,'Files completed',o,'Files Remaning')

end_time_1 = time.time()
elapsed_time_1 = end_time_1 - start_time_1

print('All Files Completed! Took {} seconds'.format(elapsed_time_1))

file_1={'RollNo':p,'Actual_Height':q,'Detected_Height':s}
df_1=pd.DataFrame(file_1)
file_2={'RollNo':p1}
df_2=pd.DataFrame(file_2)
print(df_1)
print(df_2)

'''
print('Saving Results...')

#to save results as csv files
 
file_1={'RollNo':p,'Actual_Height':q,'Detected_Height':s}
df_1=pd.DataFrame(file_1)
df_1.to_csv('L1.csv',index=False)

file_2={'RollNo':p1}
df_2=pd.DataFrame(file_2)
df_2.to_csv('EL1.csv',index=False)

print('Files Saved')
'''
