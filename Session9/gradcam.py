import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

class _BaseWrapper(object):
    def __init__(self, model):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # a set of hook function handlers

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, image):
        self.image_shape = image.shape[2:]
        self.logits = self.model(image)
        self.probs = F.softmax(self.logits, dim=1)
        return self.probs.sort(dim=1, descending=True)  # ordered results

    def backward(self, ids):
        """
        Class-specific backpropagation
        """
        one_hot = self._encode_one_hot(ids)
        self.model.zero_grad()
        self.logits.backward(gradient=one_hot, retain_graph=True)

    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()


class GradCAM(_BaseWrapper):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure 2 on page 4
    """

    def __init__(self, model, candidate_layers=None):
        super(GradCAM, self).__init__(model)
        self.fmap_pool = {}
        self.grad_pool = {}
        self.candidate_layers = candidate_layers  # list

        def save_fmaps(key):
            def forward_hook(module, input, output):
                self.fmap_pool[key] = output.detach()

            return forward_hook

        def save_grads(key):
            def backward_hook(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0].detach()

            return backward_hook

        # If any candidates are not specified, the hook is registered to all the layers.
        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(
                    module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(
                    module.register_backward_hook(save_grads(name)))

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def generate(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        weights = F.adaptive_avg_pool2d(grads, 1)

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = F.relu(gcam)
        gcam = F.interpolate(
            gcam, self.image_shape, mode="bilinear", align_corners=False
        )

        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        return gcam



class UnNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    '''
        UnNormalizes an image given its mean and standard deviation
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
    '''

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

'''
    Takes input as images, labels, device and target layers and returns model predictions and 
    Args:
    images - Image dataset
    labels - Corresponding labels
    model - Model used
    device - cuda or cpu
    target_layers- list of layers on which computation is done
    Returns:
    layers_region - 
    pred_probs - Output of model fwd prop on the images
    pred_ids - Output of model fwd prop on the images
'''


def get_gradcam(images, labels, model: torch.nn.Module, device: str, target_layers: list):

    model.to(device)

    model.eval()

    gcam = GradCAM(model=model, candidate_layers=target_layers)

    # predicted probabilities and class ids
    # Predictions of the model on the data
    pred_probs, pred_ids = gcam.forward(images)
    # actual class ids
    target_ids = labels.view(len(images), -1).to(device)

    # backward pass wrt to the actual ids
    gcam.backward(ids=target_ids)

    # we will store the layers and correspondings images activations here
    layers_region = {}

    # fetch the grad cam layers of all the images
    for target_layer in target_layers:
        # Grad-CAM generate function??
        regions = gcam.generate(target_layer=target_layer)
        layers_region[target_layer] = regions

    # we are done here, remove the hooks
    gcam.remove_hook()

    return layers_region, pred_probs, pred_ids


def plt_gradcam(gcam_layers, images, target_labels, predicted_labels, class_labels, un_normalize, paper_cmap=False):

    images = images.cpu()
    # convert BCHW to BHWC for plotting stuff

    images = images.permute(0, 2, 3, 1)
    target_labels = target_labels.cpu()

    fig, axs = plt.subplots(nrows=len(images), ncols=len(
        gcam_layers.keys())+2, figsize=((len(gcam_layers.keys()) + 2)*2, len(images)*2))
    fig.suptitle("Grad-CAM", fontsize=16)

    for image_idx, image in enumerate(images):

        # un-normalize the image
        denorm_img = un_normalize(image.permute(2, 0, 1)).permute(1, 2, 0)

        axs[image_idx, 0].text(
            0.5, 0.5, f'predicted: {class_labels[predicted_labels[image_idx][0] ]}\nactual: {class_labels[target_labels[image_idx]] }', horizontalalignment='center', verticalalignment='center', fontsize=14, )
        axs[image_idx, 0].axis('off')

        axs[image_idx, 1].imshow(
            (denorm_img.numpy() * 255).astype(np.uint8),  interpolation='bilinear')
        axs[image_idx, 1].axis('off')

        for layer_idx, layer_name in enumerate(gcam_layers.keys()):
            # gets H X W of the cam layer
            _layer = gcam_layers[layer_name][image_idx].cpu().numpy()[0]
            heatmap = 1 - _layer
            heatmap = np.uint8(255 * heatmap)
            heatmap_img = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            superimposed_img = cv2.addWeighted(
                (denorm_img.numpy() * 255).astype(np.uint8), 0.6, heatmap_img, 0.4, 0)

            axs[image_idx, layer_idx +
                2].imshow(superimposed_img, interpolation='bilinear')
            axs[image_idx, layer_idx+2].set_title(f'layer: {layer_name}')
            axs[image_idx, layer_idx+2].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, wspace=0.2, hspace=0.2)
    plt.show()
    plt.savefig('gradcam.png')


def generate_gradcam(model, test_loader, device, target_layers, mean, std, classes):

    data, target = next(iter(test_loader))

    data, target = data.to(device), target.to(device)  # Sending to Gradcam

    gcam_layers, predicted_probs, predicted_classes = get_gradcam(
        data, target, model, device, target_layers)

    # get the denomarlization function
    unorm = UnNormalize(mean=mean, std=std)

    plt_gradcam(gcam_layers=gcam_layers, images=data, target_labels=target,
                predicted_labels=predicted_classes, class_labels=classes, un_normalize=unorm)

