Step1: Run pre.py if sucessfull skip 'Step2'
Step2: Install missing libraries
Step3: Install PytesseractOCR using the link
Step4: Open CMD prompt
Step5: Run obj_det_num_extract.py and enter path when needed

copy & paste below link in the browser to download pytesseractOCR

https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-5.3.0.20221222.exe


PATH_TO_CFG="<path_to_model>/my_model/pipeline.config"
PATH_TO_CKPT="<path_to_model>/my_model/checkpoint"

PATH_TO_LABELS = '<path_to_file>/label_map.pbtxt'

pytesseract.tesseract_cmd='<path_to_.exe file>\\tesseract.exe'

server = name of server(localhost)
user = username of sql server
password = password to the sql user server
database = databas name
