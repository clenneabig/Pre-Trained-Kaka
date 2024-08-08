import timm
import torch
import torchvision.transforms as T
from torchsummary import summary
import random
from PIL import Image
from urllib.request import urlopen
import os
import pandas as pd
from wildlife_datasets import datasets, splits
from wildlife_tools.features import DeepFeatures
from wildlife_tools.data import WildlifeDataset
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.inference import KnnClassifier
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, KFold
from skimage.segmentation import mark_boundaries
from lime import lime_image



if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    if torch.cuda.is_available():
        torch.set_default_device(device)
        torch.cuda.empty_cache()




    class FirstYearKaka(datasets.DatasetFactory):
        def create_catalogue(self) -> pd.DataFrame:
            data_dir = '/home/clenneabig/Desktop/AIML591/FirstYear/Full Dataset'

            img_by_label = dict()

            for folder in os.listdir(data_dir):
                imgs = [i for i in glob.iglob(data_dir + "/" + folder + "/*")]
                img_by_label[folder] = imgs

            all_imgs = [j for i in img_by_label.values() for j in i]
            all_labels = [i for i in img_by_label for j in range(len(img_by_label[i]))]


            df = pd.DataFrame({
                'image_id': list(range(1, len(all_imgs)+1)),
                'identity': all_labels,
                'path': all_imgs,
            })
            return df
        

    #datasets.FriesianCattle2015.get_data('data/FriesianCattle2015')
    # d = datasets.FriesianCattle2015('data/FriesianCattle2015')
    # dataset = WildlifeDataset(d.df, d.root, img_load='crop_black')
    # df1 = dataset.metadata[dataset.metadata['identity'] == 1]
    # df2 = dataset.metadata[dataset.metadata['identity'] == 2]
    # df_database = pd.concat((df1.iloc[2:], df2.iloc[2:]))
    # df_query = pd.concat((df1.iloc[:2], df2.iloc[:2]))

    # print(df_database)
    # print(df_query)
        

    d = FirstYearKaka('/home/clenneabig/Desktop/AIML591/FirstYear/Full Dataset')


    transform = T.Compose([T.Resize(size=(384, 384)), 
                                T.ToTensor(), 
                                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]) 
    
    # dataset = WildlifeDataset(d.df, d.root, transform=transform, img_load='crop_black')

    size = 984

    k = 8

    idxs = list(range(size))

    labels = d.df.get('identity').tolist()

    # random.shuffle(idxs)

    fold_idx = []


    skf = StratifiedKFold(n_splits=k, shuffle=True)
    kf = KFold(n_splits=k, shuffle=True)

    # for i, (train_index, test_index) in enumerate(skf.split(idxs, labels)):
    #     fold_idx.append(test_index.tolist())

    # fold = []
    # for i in range(size):
    #     fold.append(idxs[i])
    #     if len(fold) == int(size/k):
    #         fold_idx.append(fold)
    #         fold = []


    accs = []
    preds = []
    actuals = []

    # for i in range(k):
    for i, (idx_train, idx_test) in enumerate(skf.split(idxs, labels)):
        print("Validation " + str(i + 1))

        #splitter = splits.ClosedSetSplit(0.8)
        # idx_test = fold_idx[i]
        # idx_train = []
        # for j in range(8):
        #     if i != j:
        #         idx_train = idx_train + fold_idx[j]

        #idx_train, idx_test = splitter.split(d.df)[0]
        df_database = WildlifeDataset(d.df.loc[idx_train], d.root, transform=transform, img_load='crop_black')
        np.save("/home/clenneabig/Desktop/AIML591/Pre-Trained/database_labels.npy", df_database.labels_string)
        #labels_string = np.load("/home/clenneabig/Desktop/AIML591/Pre-Trained/database_labels.npy", allow_pickle=True)
        df_query = WildlifeDataset(d.df.loc[idx_test], d.root, transform=transform, img_load='crop_black')
        np.save("/home/clenneabig/Desktop/AIML591/Pre-Trained/query_labels.npy", df_query.labels_string)
        #q_labels = np.load("/home/clenneabig/Desktop/AIML591/Pre-Trained/query_labels.npy", allow_pickle=True)

        for p in df_query.metadata.get('path').tolist():
            print(p)

        #print(df_query.metadata)

        # path = df_query.metadata.iat[0, 2]

        # print(path)



        model = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True)
        #model = model.to(device)
        #model.eval()




        extractor = DeepFeatures(model, device=device, num_workers=2)


        database = extractor(df_database)
        query = extractor(df_query)

        np.save("/home/clenneabig/Desktop/AIML591/Pre-Trained/database.npy", database)
        np.save("/home/clenneabig/Desktop/AIML591/Pre-Trained/query.npy", query)

        # database = np.load("/home/clenneabig/Desktop/AIML591/Pre-Trained/database.npy", allow_pickle=True)
        # query = np.load("/home/clenneabig/Desktop/AIML591/Pre-Trained/query.npy", allow_pickle=True)

        # print(type(database))
        # print(type(query))

        sim_func = CosineSimilarity()
        sim = sim_func(query, database)['cosine']

        # print(sim)
        # print(sim.shape)

        clas_func = KnnClassifier(k=3, database_labels=df_database.labels_string)
        pred, probs = clas_func(sim)

        #print(pred)


        count_true = 0
        preds.append(pred.tolist())
        q_labels = df_query.labels_string.tolist()
        actuals.append(q_labels)
        for i in range(len(pred)):
            #print("Prediction: " + pred[i] + " Actual: " + df_query.metadata.iat[i, 1])
            if pred[i] == q_labels[i]:
                count_true += 1

        print("Accuracy: " + str(count_true/len(pred)))
        accs.append(str(count_true/len(pred)))
        break


    # with open(r"/home/clenneabig/Desktop/AIML591/Pre-Trained/validationk3.txt", 'w') as fp:
    #     for i in range(k):
    #         fp.write(' '.join(preds[i]) + "\n")
    #         fp.write(' '.join(actuals[i]) + "\n")
    #         fp.write(accs[i] + "\n")
    #         fp.write(' \n')


    # summary(model, input_size=(3, 384, 384))
