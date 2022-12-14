import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance
import cv2



faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyes=cv2.CascadeClassifier('haarcascade_eye.xml')
def detect_faces(up_image):
	detect_img=np.array(up_image.convert('RGB'))
	new_img1=cv2.cvtColor(detect_img,1)
	# gray=cv2.cvtColor(new_img1,cv2.COLOR_BGR2GRAY)
	faces=faceDetect.detectMultiScale(new_img1,1.3,5)
	for x,y,w,h in faces:
		cv2.rectangle(new_img1,(x,y),(x+w,y+h),(255,255,0),2)
	return new_img1,faces
def detect_eye(up_image):
	detect_img=np.array(up_image.convert('RGB'))
	new_img1=cv2.cvtColor(detect_img,1)
	# gray=cv2.cvtColor(new_img1,cv2.COLOR_BGR2GRAY)
	faces=eyes.detectMultiScale(new_img1,1.3,5)
	for x,y,w,h in faces:
		cv2.rectangle(new_img1,(x,y),(x+w,y+h),(255,255,0),2)
	return new_img1,faces





def main():
	
	st.title("Face Detection App")
	
	
	

	
	
	img_file=st.file_uploader("Upload File",type=['png','jpg','jpeg'])
	if img_file is not None:
		up_image=Image.open(img_file)
		st.image(up_image)
		
		
	if st.button("Process"):
			
		result_img,result_faces=detect_faces(up_image)
		st.image(result_img)
		st.success("Found {} faces".format(len(result_faces)))
			
			
	

if __name__=='__main__':
	main()