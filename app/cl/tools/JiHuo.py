import argparse,os,sys,subprocess
from co6co.utils.log import progress_bar

def execute_command(command):
    """
    执行执行 命令 比如：command = 'netstat -ano | findstr "LISTENING"'
    """
    try:
        result = subprocess.check_output(command, shell=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        return None
def execute_command2(command):
    """
    执行执行 命令 比如：command = 'netstat -ano | findstr "LISTENING"'
    """
    try:
        # 使用 subprocess.run 执行命令，stdout 设置为 PIPE 以捕获输出
        # result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True)
        # result = subprocess.run( command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        result = subprocess.check_output(command,shell=True,stderr=subprocess.STDOUT).decode()
        return True 
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
        

def jihuo(key_file,kms_file:str):
    """
    slmgr -skms {s}
    slmgr -ipk {k}
    slmgr -ato
    """
    command0 = 'cscript //B "c:\\windows\\system32\\slmgr.vbs" /skms {key}'
    command2 = 'cscript //B "c:\\windows\\system32\\slmgr.vbs" /ipk {key}'
    command3 = 'cscript //B "c:\\windows\\system32\\slmgr.vbs" /ato'
    with open(key_file, 'r') as file,open(kms_file,'r')  as k_file:
        keys = file.read().splitlines(False)
        kms=k_file.read().splitlines(False)
        isSucc=False
        index_i=0
        index_j=0
        total=len(kms)*len(keys)
        for s in kms:
            index_i+=1
            if isSucc:
                break
            if not s:
                print(f"无效的：s->{s}")
                continue
            command = command0.format(key=s)
            print("设置KMS->",s)
            r=execute_command(command) #cscript //B "c:\\windows\\system32\\slmgr.vbs" /skms {key}
            if not r:
                continue   
            for k in keys:
                index_j+=1
                #invalid
                progress_bar(index_i*index_j/total,f"进度{index_i}，{index_j}")
                if not k or not s: 
                    continue
                print("设置KEY->",k)
                command = command2.format(key=k)
                r=execute_command(command)
                if not r:
                    continue  
                command = command3
                print("ato..")
                r=execute_command(command)
                if r:
                    print("success")
                    isSucc=True
                    break

def getCurrent():
     if "python.exe" in sys.executable:
            curr=__file__
     else:
        curr=sys.executable
     return os.path.dirname(os.path.abspath(curr))
     
        
    
if __name__ == "__main__":
    # nuitka --standalone --onefile --windows-icon-from-ico=c:\Users\Administrator\Pictures\cardano.ico .\tools\ftpCheck.py
    parser = argparse.ArgumentParser(description="JiHUO")
 
    print("当前文件：", __file__,getCurrent())
    curr=getCurrent()
    parser.add_argument("-k", "--key_file", default=f"{curr}\\keys.txt",  type=str, help=f"keys Files, default: {curr}\\keys.txt")
    parser.add_argument("-s", "--kms_file", default=f"{curr}\\kms.txt", type=str, help=f"kms 服务器,default:{curr}\\kms.txt")  

    args=parser.parse_args()
    kfile=args.key_file
    sfile=args.kms_file
    if os.path.exists(kfile) and os.path.exists(sfile): 
        jihuo(kfile,sfile)
    else:
        parser.print_help()