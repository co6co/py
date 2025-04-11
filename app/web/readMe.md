# 创建初
目的：管理本地计算机,可进行远程设置IP、启停服务、创建任务等操作

1. opencv-python
  
   cap = cv2.VideoCapture(video_path)
		if not cap.isOpened(): #  4.10.0.82 ok 但是 4.11.xx  xx 不记得是什么版本
							   #  pyinstaller 打包后不能 opend
		
			pass