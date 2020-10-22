import cv2
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import model_from_json
import pyautogui          

IMG_SIZE = 128

# pyautogui
screenWidth, screenHeight = pyautogui.size()
pyautogui.moveTo(screenWidth/2, screenHeight/2)

#function that form our tasks and form our actions
def performTask(action):
    move_num = 10
    scroll = 40
    currentMouseX, currentMouseY = pyautogui.position()
    count_action  = 0
    
    if action == "pause":
        pyautogui.press('space')
        
        
    elif action == "play":
        pyautogui.press('space')
        
        
    elif action == "prev track":
        pyautogui.press('left')
        

    elif action == "next track":
        pyautogui.press('right')
        

    elif action == "volume up":
        pyautogui.press('up')

    elif action == "volume down":
        pyautogui.press('down')
    


def loadModel():
	json_file = open('try.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("try.h5")
	print("Loaded model from disk")
	return loaded_model

isBgCaptured           = False    # bool, whether the background captured
cap_region_x_begin     = 0.0      # start point/total width
cap_region_y_end       = 0.5      # start point/total width
font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 1
fontColor              = (255,0,255)
lineType               = 2
bottomLeftCornerOfText = (55,25)



cap = cv2.VideoCapture(0)
ret,first_frame = cap.read()
model = loadModel()
decision = ['pause', 'play','prev track',  'next track', 'volume up',  'volume down','no gesture']

while ret:
	X = []
	ret,frame = cap.read()
	cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),(int(0.5* frame.shape[1]), int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
	img = frame[0:int(cap_region_y_end * frame.shape[0]),int(cap_region_x_begin * frame.shape[1]):int(0.5* frame.shape[1])]
	img = cv2.resize(img,(128,128))
	X.append(img)
	X = np.array(X)
	X = X/255.
	pred = model.predict(X)
	pred = np.argmax(pred,axis = 1)
	cv2.putText(frame,decision[pred[0]],bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
	performTask(decision[pred[0]])
	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()