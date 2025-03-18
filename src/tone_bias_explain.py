
"""
Note that this code (specifically the libraries) are not compatible with the PyTorch MPS installed.
So do not use "conda activate pytorch_test".
Need to use "conda activate explain".
The "explain" is a straight forward conda install of captum, (torch), torchvision, and shap.
  1013  conda install captum
  1015  conda install torchvision
  1017  conda install shap
"""

import torch
import torch.nn.functional as F

from PIL import Image

import os
import json
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import torchvision
from torchvision import models
from torchvision import transforms

from captum.attr import Saliency
from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import LRP
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
from captum.attr._utils.lrp_rules import EpsilonRule, GammaRule, Alpha1_Beta0_Rule


# Local imports
from tone_bias_dataset import HibaDataset
from tone_bias_dataset import Rescale
from tone_bias_dataset import ToTensor
import tone_bias_dataset
import pandas as pd
import matplotlib.pyplot as plt

def main():

    experiment_dir = "/Users/james/Documents/skin-image-analysis/experiments/balanced_2024-09-21_00-38-39"
    train_df_name  = os.path.join(experiment_dir, "session_train.csv")
    test_df_name   = os.path.join(experiment_dir, "session_test.csv")
    model_name     = os.path.join(experiment_dir, "session_model.pth")

    #model = "/Users/james/Documents/skin-image-analysis/experiments/balanced_2024-09-21_00-38-39/session_model.pth"
    #imageDir = '/Users/james/Documents/skin-image-analysis/tone/ISIC_0079358.jpg'

    #modelDir = "I:/My Drive/Higher Study/Collaboration/Dr James Pope/AI Explain/session_model.pth"
    #imageDir = 'I:/My Drive/Higher Study/Collaboration/Dr James Pope/AI Explain/Data/ISIC_0024329.jpg'

    root_dir_name = '../tone'
    test_df = pd.read_csv(test_df_name)
    class_names = ['benign', 'malignant']

    test_dataset = HibaDataset(  test_df,
                            class_names,
                            root_dir=root_dir_name,
                            transform=torchvision.transforms.Compose([
                                Rescale((224, 224)),
                                # RandomCrop(224),
                                ToTensor()
                            ])
                            )

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=2,
                                              shuffle=False,
                                              num_workers=10)




    device = 'cpu'

    model = torch.load(model_name, weights_only=False)
    model = model.eval()
    # Model was trained and saved (i.e. currently set to use) 'mps' backend
    # Do not believe we need 'mps' acceleration for explainability analysis, so changing to 'cpu'
    model = model.to(device)


    batch_number = 0
    for batch in test_loader:
        images, labels, indexes = batch
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        #print(f"BATCH: {outputs.data}")
        #print(f" TYPE: {type(outputs.data)}")

        # NB: outputs is batch object with predictions in the data
        #     the predictions are log probabilities, e.g.
        #     [-0.1054, -2.3026], noting 0 is larger
        #     [   0.90,    0.10], to get probabilities take e^x
        softmax_probs = torch.exp(outputs.data)

        #_, predicted = torch.max(outputs.data, 1)
        max_values, max_indices = torch.max(softmax_probs, 1)

        #print(f"BATCH: {softmax_probs} OTHER {max_values}")
        integrated_gradients = IntegratedGradients(model)
        saliency = Saliency(model)

        print(f"BATCH {batch_number} / {len(test_loader)}")
        for i in range( len(labels) ):
            index  = indexes[i]
            output = outputs[i]

            transformed_image = images[i]
            # I think image_np and transformed_image are the same???
            image_np, label, idx = test_dataset[index]
            instance = test_dataset.lookup_path( int(index) )
            print(f"Same index idx={idx}  index={int(index)}")
            #print(f"Same object {id(image_np) == id(transformed_image)}")
            #print(f"Same object {image_np is transformed_image}")

            print(f"IMAGE {instance['file_path']} {image_np.shape}")
            print(f"TRANS {transformed_image.shape}")

            predict_idx= max_indices[i]
            label_idx  = labels[i]

            true_label = class_names[label_idx]
            pred_label = class_names[predict_idx]

            print(f"    batch [{i}] dataset [{index}]:  predict={pred_label}  true={true_label}")

            transformed_image = transformed_image.unsqueeze(0)  # Add batch dimension

            # attributions_ig = integrated_gradients.attribute(transformed_image, target=predict_idx, n_steps=200)
            attributions_ig = integrated_gradients.attribute(transformed_image, target=predict_idx, n_steps=200)

            # print(f"attributions_ig          len = {len(attributions_ig)}")

            # Need to do two things, get rid of batch dimension (with squeeze)
            # Need to change dimensions from (C,H,W) to (H,W,C)
            transposed_attribution = np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
            transposed_image = np.transpose(transformed_image.squeeze().cpu().detach().numpy(), (1, 2, 0))


            # Custom color map
            default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                             [(0, '#ffffff'),
                                                              (0.25, '#000000'),
                                                              (1, '#000000')], N=256)
            # attribution: Numpy array corresponding to attributions to be visualized.
            #              Shape must be (H, W, C), with channels as the last dimension.
            #
            # _ = viz.visualize_image_attr(transposed_attribution, transposed_image, method='heat_map',
            #         cmap=default_cmap, show_colorbar=True, sign='positive', outlier_perc=1)

            # _ = viz.visualize_image_attr_multiple(
            #     transposed_attribution,
            #     transposed_image,
            #     ["original_image", "heat_map"],
            #     ["all", "absolute_value"],
            #     cmap=default_cmap,
            #     show_colorbar=True)
            #plt.title = f"Image {instance['file_path']} true={true_label} predict={pred_label}"
            _ = viz.visualize_image_attr_multiple(
                transposed_attribution,
                transposed_image,
                ["original_image", "heat_map", "blended_heat_map"],
                ["all", "absolute_value", "positive"],
                cmap=default_cmap,
                alpha_overlay=0.7,
                show_colorbar=True)


            # I think this approach is easier to "explain" but crashes for some of the images.
            # nt = NoiseTunnel(saliency)
            # smooth_attr = nt.attribute(transformed_image, target=predict_idx, nt_samples=10, nt_type='smoothgrad')
            # smooth_attr = np.transpose(smooth_attr.squeeze().cpu().detach().numpy(), (1, 2, 0))
            #
            # _ = viz.visualize_image_attr_multiple(
            #     smooth_attr,
            #     transposed_image,
            #     ["original_image", "heat_map", "blended_heat_map"],
            #     ["all", "absolute_value", "positive"],
            #     cmap='hot',
            #     alpha_overlay=0.1,
            #     show_colorbar=True)


            # fig, axs = plt.subplots(1, 2, figsize=(15, 5))
            #
            # # Original image
            # axs[0].imshow(trans_images)
            # axs[0].set_title('Input Image')
            # axs[0].axis('off')
            #
            # axs[1].imshow(trans_attributions, cmap=default_cmap)
            # axs[1].set_title('IntegratedGradients Map')
            # axs[1].axis('off')

            # plt.show()

        if batch_number >=2:
            break



        batch_number += 1

        #img = Image.open('/Users/james/Documents/skin-image-analysis/tone/ISIC_0079358.jpg')
        #image_np, label, idx = dataset[0] # calls __getitem__, passing through image transforms



if __name__ == '__main__':
    main()