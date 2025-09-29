import os
import sys
import argparse
import torch.onnx
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..'))
from torchkit.backbone import get_model
from torchkit.util.utils import l2_norm


parser = argparse.ArgumentParser(description='export pytorch model to onnx')
parser.add_argument('--ckpt_path', default=None, type=str, required=True, help='')
parser.add_argument('--onnx_name', default=None, type=str, required=True, help='')
parser.add_argument('--model_name', default=None, type=str, required=True, help='') 
args = parser.parse_args()


def main():
    net = get_model(args.model_name)
    input_size = [112, 112]
    torch_model = net(input_size)
    if not os.path.isfile(args.ckpt_path):
        print("Invalid ckpt path: %s" % args.ckpt_path)
        return
    torch_model.load_state_dict(torch.load(args.ckpt_path, weights_only=True))
    torch_model.eval()
    
    batch_size = 1
    x = torch.randn(batch_size, 3, input_size[0], input_size[1], requires_grad=True)
    torch_out = torch_model(x)
    torch_out = l2_norm(torch_out)
    torch.onnx.export(torch_model,
                      x,
                      "%s.onnx" % args.onnx_name,
                      export_params=True,
                      opset_version=10,
                      do_constant_folding=True,
                      input_names = ['input'],
                      output_names = ['output'])
    print("finished")


if __name__ == '__main__':
    main()
