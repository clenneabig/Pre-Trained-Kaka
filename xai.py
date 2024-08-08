from wildlife_tools.data import WildlifeDataset
from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.inference import KnnClassifier
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries
import torch
import torchvision.transforms as T
import numpy as np
import pandas as pd
import timm
import os
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage

def batch_predict(images):
    global l
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    all_imgs = []
    for i in range(len(images)):
        path = "/home/clenneabig/Desktop/AIML591/Pre-Trained/images_array/image" + str(i) + ".npy"
        all_imgs.append(path)
        np.save(path, images[i])

    df = pd.DataFrame({
        'image_id': list(range(1, len(all_imgs)+1)),
        'identity': [l for i in range(len(all_imgs))],
        'path': all_imgs,
    })

    transform = T.Compose([T.Resize(size=(384, 384)), 
                                T.ToTensor(), 
                                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]) 

    df_images = WildlifeDataset(df, transform=transform)

    model = timm.create_model("hf-hub:BVRA/MegaDescriptor-L-384", pretrained=True)

    extractor = DeepFeatures(model, device=device, num_workers=2)

    database = np.load("/home/clenneabig/Desktop/AIML591/Pre-Trained/database.npy", allow_pickle=True)
    query = extractor(df_images)

    sim_func = CosineSimilarity()
    sim = sim_func(query, database)['cosine']

    labels_string = np.load("/home/clenneabig/Desktop/AIML591/Pre-Trained/database_labels.npy", allow_pickle=True)

    clas_func = KnnClassifier(k=3, database_labels=labels_string)
    pred, probs = clas_func(sim)

    return np.array(probs)




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


    def get_image(path):
        with open(os.path.abspath(path), 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB') 
            
    paths = []

    with open(r"/home/clenneabig/Desktop/AIML591/Pre-Trained/query_paths.txt", 'r') as fp:
        for line in fp:
            splits = line.split('/')
            paths.append((splits[7], splits[8][:-1]))
    

    for label, img in paths:
        imgA = get_image("/home/clenneabig/Desktop/AIML591/FirstYear/Full Dataset/" + label + "/" + img)

        transform = T.Compose([T.Resize(size=(384, 384))]) 

        global l
        l = label
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(np.array(transform(imgA)), 
                                            batch_predict, # classification function
                                            top_labels=3, 
                                            hide_color=0,
                                            batch_size=128, 
                                            num_samples=1000)
        
        #print(explanation.local_exp)

        #Select the same class explained on the figures above.
        ind =  explanation.top_labels[0]

        #Map each explanation weight to the corresponding superpixel
        dict_heatmap = dict(explanation.local_exp[ind])
        heatmap = np.vectorize(dict_heatmap.get)(explanation.segments) 

        np.save("/home/clenneabig/Desktop/AIML591/Pre-Trained/explanations/heatmap/" + label + "/" + img + "_heatmap.npy", heatmap)
        

        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=False)
        img_boundry1 = mark_boundaries(temp/255.0, mask)
        Image.fromarray((img_boundry1 * 255).astype(np.uint8)).save("/home/clenneabig/Desktop/AIML591/Pre-Trained/explanations/border/" + label + "/" + img + "_border.jpeg")
        #plt.imshow(img_boundry1)

        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
        img_boundry2 = mark_boundaries(temp/255.0, mask)
        Image.fromarray((img_boundry2 * 255).astype(np.uint8)).save("/home/clenneabig/Desktop/AIML591/Pre-Trained/explanations/posneg/" + label + "/" + img + "_posneg.jpeg")
        #plt.imshow(img_boundry2)
