import torchvision.models.segmentation as seg_models
from PIL import Image
import torch
import numpy as np
import cv2
import os
import torchvision.transforms as T
# transformations needed

# Aux
dirname = os.path.dirname(__file__)

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image

    # calcs ratios
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation = inter)
    return resized


def write_image(path, img):
    img = cv2.convertScaleAbs(img, alpha=(255.0))
    cv2.imwrite(path, img)


# Labels + helper function
def decode_segmap(image, source, nc=21):
    '''
    label_colors = np.array([(0, 0, 0),
    (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0),
    (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0),
    (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (192, 128, 128),
    (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)])
    '''
    # 0=background
    # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
    # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
    # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person  
    # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor   

    label_colors = np.array([(0, 0, 0),
        (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
        (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
        (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
        (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    # label 15 = person
    for l in range(0, nc):
        if l == 15:
            idx = image == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)

    # Load the foreground input image
    foreground = cv2.imread(source)

    # Change the color of foreground image to RGB
    # and resize image to match shape of R-band in RGB output map
    foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)
    foreground = cv2.resize(foreground, (r.shape[1], r.shape[0]))

    # Create a background array to hold white pixels
    # with the same size as RGB output map
    background = 255 * np.ones_like(rgb).astype(np.uint8)

    # Convert uint8 to float
    foreground = foreground.astype(float)
    background = background.astype(float)

    # Create a binary mask of the RGB output map using the threshold value 0
    th, alpha = cv2.threshold(np.array(rgb), 0, 255, cv2.THRESH_BINARY)

    # Apply a slight blur to the mask to soften edges
    alpha = cv2.GaussianBlur(alpha, (7, 7), 0)

    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = alpha.astype(float)/255

    # Multiply the foreground with the alpha matte
    foreground = cv2.multiply(alpha, foreground)

    # Multiply the background with ( 1 - alpha )
    background = cv2.multiply(1.0 - alpha, background)

    # Add the masked foreground and background
    outImage = cv2.add(foreground, background)

    # Return a normalized output image for display
    return outImage/255


def segment(net, path, dev='cpu'):
    img = Image.open(path)

    # Comment Resize and CenterCrop for better inference results
    trf = T.Compose([T.Resize(450),
                     # T.CenterCrop(224),
                     T.ToTensor(),
                     T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

    inp = trf(img).unsqueeze(0).to(dev)
    out = net.to(dev)(inp)['out']

    om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    rgb = decode_segmap(om, path)

    # Resize back to orig
    w, h = img.size[:]
    rgb = image_resize(rgb, width=w, height=h, inter=cv2.INTER_LINEAR)
    return rgb


# dlab = seg_models.deeplabv3_resnet101(pretrained=True).eval()
fcnn = seg_models.fcn_resnet101(pretrained=True).eval()

# segment(dlab, os.path.join(dirname, 'images/front1.jpg'), dev='cpu')

output_img = segment(fcnn, os.path.join(dirname, 'images/front1.jpg'), dev='cpu')

cv2.imshow('Sin fondo', output_img)
write_image(os.path.join(dirname, 'filtered_images/result_front.jpg'), output_img)

if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()
