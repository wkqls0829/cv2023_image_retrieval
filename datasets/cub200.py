from datasets.basic_dataset_scaffold import BaseDataset
import os
import numpy as np

def Give(opt, datapath):

    # Load Dataset
    image_sourcepath  = datapath+'/images'  # img class 별로 있는 폴더
    image_classes     = sorted([x for x in os.listdir(image_sourcepath) if '._' not in x], key=lambda x: int(x.split('.')[0])) # classes를 numbering 기준으로 정렬 001.Black_footed_Albatross
    total_conversion  = {int(x.split('.')[0])-1:x.split('.')[-1] for x in image_classes} # dict = {0:Black_footed_Albatross,  } 0번 부터 시작
    image_list        = {int(key.split('.')[0])-1:sorted([image_sourcepath+'/'+key+'/'+x for x in os.listdir(image_sourcepath+'/'+key) if '._' not in x]) for key in image_classes} 
    # img_list = {0: [source_path/images/001.Black_footed_Albatross/1.jpg, ..], 1:[], ...}
    image_list        = [[(key,img_path) for img_path in image_list[key]] for key in image_list.keys()] # [(0, source_path/images/001.Black_footed_Albatross/1.jpg), (0, 2.jpg)], [(1,1.jpg), ]
    
    
    # Truncate Dataset for LT Ratio
    cls_num = len(image_list)
    img_max = len(image_list[0]) - 10
    img_num_per_cls = get_img_num_per_cls(img_max, cls_num, opt.imb_factor) # img number를 점점 줄어들게 list
    opt.img_num_per_cls = img_num_per_cls
    test_image_list = [images[-10:] for images in image_list] # 각 class에 대해서 마지막 10장을 test로 사용
    print(f"spliting test data into {[len(im) for im in test_image_list]} number of datasets")
    image_list = [images[:img_num] for images, img_num in zip(image_list, img_num_per_cls)] # image_list = [[], [],...] lt로 필요한 길이만큼 잘라줌 ### 요걸 plot 해줘야 겠구만 ###
    print(f"spliting train data into {[len(im) for im in image_list]} number of datasets")  
    test_image_list        = [x for y in test_image_list for x in y]  # [[(), (), ()], []... ] --> [(), (), ()....]
    image_list        = [x for y in image_list for x in y]
    

    ### Dictionary of structure class:list_of_samples_with_said_class
    image_dict    = {}
    for key, img_path in image_list:
        if not key in image_dict.keys():
            image_dict[key] = []
        image_dict[key].append(img_path)   # {0: [], 1: [], ...}

    test_image_dict    = {}
    for key, img_path in test_image_list:
        if not key in test_image_dict.keys():
            test_image_dict[key] = []
        test_image_dict[key].append(img_path)

    ### Use the first half of the sorted data as training and the second half as test set
    keys = sorted(list(image_dict.keys()))
    test_keys = sorted(list(test_image_dict.keys()))
    train = keys
    test = test_keys
    ## train,test      = keys[:len(keys)//2], keys[len(keys)//2:]

    ### If required, split the training data into a train/val setup either by or per class.
    if opt.use_tv_split:
        if not opt.tv_split_by_samples:
            train_val_split = int(len(train)*opt.tv_split_perc)
            train, val      = train[:train_val_split], train[train_val_split:]
            ###
            train_image_dict = {i:image_dict[key] for i,key in enumerate(train)}
            val_image_dict   = {i:image_dict[key] for i,key in enumerate(val)}
            test_image_dict  = {i:image_dict[key] for i,key in enumerate(test)}
        else:
            val = train
            train_image_dict, val_image_dict = {},{}
            for key in train:
                train_ixs   = np.array(list(set(np.round(np.linspace(0,len(image_dict[key])-1,int(len(image_dict[key])*opt.tv_split_perc)))))).astype(int)
                val_ixs     = np.array([x for x in range(len(image_dict[key])) if x not in train_ixs])
                train_image_dict[key] = np.array(image_dict[key])[train_ixs]
                val_image_dict[key]   = np.array(image_dict[key])[val_ixs]
        val_dataset    = BaseDataset(val_image_dict, opt, is_validation=True)
        val_conversion = {i:total_conversion[key] for i,key in enumerate(val)}
        ###
        val_dataset.conversion   = val_conversion
    else:
        train_image_dict = {key:image_dict[key] for key in train}
        val_image_dict   = None
        val_dataset      = None

    ###
    train_conversion = {i:total_conversion[key] for i,key in enumerate(train)}
    test_conversion  = {i:total_conversion[key] for i,key in enumerate(test)}

    ###
    test_image_dict = {key:test_image_dict[key] for key in test}

    ###
    print('\nDataset Setup:\nUsing Train-Val Split: {0}\n#Classes: Train ({1}) | Val ({2}) | Test ({3})\n'.format(opt.use_tv_split, len(train_image_dict), len(val_image_dict) if val_image_dict else 'X', len(test_image_dict)))

    ###
    train_dataset       = BaseDataset(train_image_dict, opt)
    test_dataset        = BaseDataset(test_image_dict,  opt, is_validation=True)
    eval_dataset        = BaseDataset(train_image_dict, opt, is_validation=True)
    eval_train_dataset  = BaseDataset(train_image_dict, opt, is_validation=False)
    train_dataset.conversion       = train_conversion
    test_dataset.conversion        = test_conversion
    eval_dataset.conversion        = test_conversion
    eval_train_dataset.conversion  = train_conversion

    return {'training':train_dataset, 'validation':val_dataset, 'testing':test_dataset, 'evaluation':eval_dataset, 'evaluation_train':eval_train_dataset}


def get_img_num_per_cls(img_max, cls_num, imb_factor): # imb = 1, 0.5, 0.1               
    img_num_per_cls = []
    for cls_idx in range(cls_num):
        num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
        img_num_per_cls.append(int(num))

    return img_num_per_cls

