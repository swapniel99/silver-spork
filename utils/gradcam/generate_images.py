from torch.nn import functional as F
import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from ezio.utils.gradcam.grad_cam import GradCAM
from ezio.data.data_transforms.session_9.transformations import denormalize_image



def gradcam_features(model, target_layers, images, labels, device):
    # Move images and model to device and set to eval mode
    # map input to device
    images = images.to(device)
    model.to(device)
    model.eval()

    # Gradcam output/features
    g_cam = GradCAM(model=model, candidate_layers=target_layers)

    # Actual labels
    target_labels = torch.LongTensor(labels).view(len(images), -1).to(device)

    # Generate predicted labels with their probabilities
    predicted_probs, predicted_labels = g_cam.forward(images)

    # GradCam Backpropagation
    g_cam.backward(ids=target_labels)

    # Dictionary to store the gradcam outputs
    gradcam_outputs = dict()

    # Generating outputs
    for layer in target_layers:
        # Gradcam generate function
        layer_output = g_cam.generate(target_layer=layer)

        # Store the output
        gradcam_outputs[layer] = layer_output

    # Remove the hook
    g_cam.remove_hook()

    return predicted_probs, predicted_labels, gradcam_outputs


def plot_grid(gcam_features, images, target_labels, predicted_labels, label_texts, image_size):
    # Generate grid parameters
    rows = len(images)
    cols = len(gcam_features) + 2
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(14,32))

    for image_index, image in enumerate(images):
        image = np.uint8(255*denormalize_image(image.view(image_size)))
        axs[image_index, 0].axis('off')

        axs[image_index, 1].imshow(image, interpolation='bilinear')
        axs[image_index, 1].set_title(f'Actual Label: {label_texts[target_labels[image_index]]}'
                                     f'\nPredicted Label: {label_texts[predicted_labels[image_index][0]]}')
        axs[image_index, 1].axis('off')

        for target_index, layer in enumerate(gcam_features):
            current_layer = gcam_features[layer][image_index].cpu().numpy()[0]
            heatmap = 1 - current_layer  # reverse the color map
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img = cv2.resize(cv2.addWeighted(image, 0.5, heatmap, 0.5, 0), (128, 128))

            axs[image_index, target_index+2].imshow(superimposed_img, interpolation='bilinear')
            axs[image_index, target_index+2].set_title(f'layer: {layer}')
            axs[image_index, target_index+2].axis('off')

    plt.show()