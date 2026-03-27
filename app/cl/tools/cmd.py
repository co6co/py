p="F:\\cygwin64\\bin\\ssh-keygen.exe" 
import os
with open("./dist/pwdgen.txt","r") as f:
    data=f.readlines()
    for item in data:
        item=item.strip()
        # 执行ping命令，但无法获取输出结果
        exe=p+" -y -f C:\\Users\\Administrator\\.ssh\\id_rsa -P "+item 
        data=os.system(exe)
        if data==0:
            print("成功==>",item)
            break
         
d