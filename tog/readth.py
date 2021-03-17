import torch

# dict = torch.load('model_weights/FRCNN.pth')
# print(dict['model'].keys())
# dict1 = torch.load('model_weights/vgg16_faster_rcnn_iter_110000.pth')
# print(dict['model'].keys())


# model_dict = model.state_dict()
# # print(model_dict)
pretrained_dict = torch.load("/yzc/reid_testpcb/se_resnet50-ce0d4300.pth")
keys = []
for k, v in pretrained_dict.items():
       keys.append(k)
i = 0
# for k, v in model_dict.items():
#     if v.size() == pretrained_dict[keys[i]].size():
#          model_dict[k] = pretrained_dict[keys[i]]
#          #print(model_dict[k])
#          i = i + 1
# model.load_state_dict(model_dict)