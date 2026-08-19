import uuid
f=0
for i in range(100000): 
    s=uuid.uuid4()
    if "3529ac8c-ef30-4dd7-ae12-cc34cb0ec153" == str(s):
        i==0
        print(s)
        f=1
    if f==1:
        print(s)
 