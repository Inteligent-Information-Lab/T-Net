import torch


dict = torch.load('SOTS_haze_best_4_3',map_location=torch.device('cpu'))
dict1 =dict
keyname= []
keyname1=[]
for key in dict:
    keyname.append(key)
    media = key.split('.', 2);
    media[1] ='TNet'
    key1 = '.'.join(media)
    keyname1.append(key1)

for i  in range(len(keyname)):
    dict[keyname1[i]]=dict.pop(keyname[i])

print()
