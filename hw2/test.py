import re

a = re.sub(r'hello', 'i love the', 'hello world')
print(a)

s = '321asdf12 asdff21 wewreqr 1 asdfrew asfdfasd32r    32 321 fads  3142   ` fdas12        r       3r434 12 #$#$ 5  %%  34 1 F SA'
s = s.split()
print(s)
s = re.sub(r'(\d+)', '', s)
s = re.sub(r'(^a-zA-Z)', '', s)

print(s)