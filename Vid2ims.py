import cv2
import pathlib
import numpy as np
########################################

#Video to images
def video2ims(video_name, outfold, dim=(512,512) ):
			
	cap = cv2.VideoCapture(video_name)

	pathlib.Path(outfold + "/images/").mkdir(exist_ok=True, parents=True)
	
	frame_counter = 0
	print("# ", frame_counter)

	while cap.isOpened():
		ret, img = cap.read()

		if not ret:
			print("STream end, exiting...")
			break
		
		#img 1920x1080 -> 512,512
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = cv2.resize(img, dsize=(0,0), fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
		s1, s2 = img.shape
		s1, s2 = s1-dim[0], s2-dim[0]
		s1, s2 = int(s1/2), int(s2/2)
		img = img[ s1:-s1, s2:-s2]
		img = cv2.resize(img, dsize=dim, interpolation = cv2.INTER_CUBIC) 
	
			
		if frame_counter%10 == 0:
			cv2.imwrite(outfold + "/images/" + str(frame_counter) + ".png", img)
			print("# ", frame_counter, " Done")

		frame_counter = frame_counter+1
		
	cap.release()
	print("END.")

if __name__ == "__main__":
		
	#Video in & out
	video_name = "/media/aswin/336163b5-742f-40f5-9e55-c18e519d29f3/McMaster_box/box/REAL_DATA_test/my_video-1.mkv"
	vid_outfold = "/media/aswin/336163b5-742f-40f5-9e55-c18e519d29f3/McMaster_box/box/REAL_DATA_test/out/"
	
	video2ims(video_name, vid_outfold)
