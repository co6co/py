from co6co.enums import Base_EC_Enum,Base_Enum

class b2(Base_Enum):
    name2="key" ,"value"

class CC(Base_EC_Enum):
    c="k","中文",1
    c1="k","中文",1

 

print(CC.c.key2enum("k"))