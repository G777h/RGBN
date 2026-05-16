import os
import sys
import cv2
import numpy as np
import torch

DIR = os.path.abspath(os.path.dirname(__file__))
print(DIR)
sys.path.append(os.path.abspath(os.path.dirname(DIR)))
import faulthandler

from config.args import test_options
from config.config import MLIC_model_config, model_config
from PIL import Image, ImageFile
from testing.tester_concat import TesterConcat
from testing.tester_master import TesterMaster
from testing.tester_single import TesterSingle
from testing.tester_united import TesterUnited

faulthandler.enable()
torch.set_default_tensor_type('torch.cuda.FloatTensor')

gate_counter = 0

def get_gate_and_save(scale_name, save_dir="visualizations/cmgm_gates"):
    
    os.makedirs(save_dir, exist_ok=True)
    
    def hook(model, input, output):
        global gate_counter
       
        feat_tensor = output.detach().cpu()

       
        for b in range(feat_tensor.size(0)):
            feature = feat_tensor[b] 
            
            feat_map = torch.mean(feature, dim=0).numpy()

            feat_map = feat_map - np.min(feat_map)
            feat_map = feat_map / (np.max(feat_map) + 1e-8)
            feat_map = np.uint8(255 * feat_map)

            heatmap = cv2.applyColorMap(feat_map, cv2.COLORMAP_JET)

            save_path = os.path.join(save_dir, f"{scale_name}_img{gate_counter+b}.png")
            cv2.imwrite(save_path, heatmap)
            print(f"Save in {save_path}")

        gate_counter += feat_tensor.size(0)
    return hook

def main(argv):
    torch.backends.cudnn.deterministic = True
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = None

    args = test_options(argv)
    if args.model.find("MLIC")!=-1:
        config = MLIC_model_config()
    else:
        config = model_config()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    print(args)
    if args.channel in [4, 6]:
        if args.model.find("cat") != -1:
            print("TesterConcat")
            tester = TesterConcat(args, config)
        else:
            print("TesterUnited")
            tester = TesterUnited(args, config)
            if hasattr(tester, 'net'):
              print("Hooking...")
              tester.net.g_a.rgb_analysis_transform[3].register_forward_hook(
                 get_gate_and_save('before')
              )
              tester.net.g_a.rgb_analysis_transform[4].dwconv1.register_forward_hook(
                 get_gate_and_save('scale1_small_field')
              )
              tester.net.g_a.rgb_analysis_transform[4].dwconv2.register_forward_hook(
                 get_gate_and_save('scale2_medium_field')
              )
              tester.net.g_a.rgb_analysis_transform[4].dwconv3.register_forward_hook(
                 get_gate_and_save('scale3_large_field')
              )
              tester.net.g_a.rgb_analysis_transform[5].register_forward_hook(
                 get_gate_and_save('after')
              )

    else:
        if args.model.find("master") != -1:
            print("TesterMaster")
            tester = TesterMaster(args, config)
        else:
            print("TesterSingle")
            tester = TesterSingle(args, config)
    tester.test_model(padding_mode="replicate0", padding=True)

if __name__ == "__main__":
    main(sys.argv[1:])
