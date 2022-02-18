import argparse
import cv2
import numpy as np
import onnxruntime
import torchvision.transforms as transforms


parser = argparse.ArgumentParser(description='onnx inference')
parser.add_argument('--onnx_path', default=None, type=str, required=True, help='')
args = parser.parse_args()


def l2_norm(x):
    """ l2 normalize
    """
    output = x / np.linalg.norm(x)
    return output


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def get_test_transform():
    test_transform = transforms.Compose([    
            transforms.ToTensor(),    
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    return test_transform


def main():
    img = cv2.imread("brucelee.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = get_test_transform()(img)
    img = img.unsqueeze_(0)
    print(img.shape)
    onnx_path = args.onnx_path
    session = onnxruntime.InferenceSession(onnx_path)
    inputs = {session.get_inputs()[0].name: to_numpy(img)}
    outs = session.run(None, inputs)[0]
    print(outs.shape)
    outs = l2_norm(outs).squeeze()
    ss  = ""
    for x in outs:
        ss = ss +  "%.6f " % x
    print(ss)


if __name__ == '__main__':
    main()
