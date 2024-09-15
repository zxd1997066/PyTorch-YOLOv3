from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.augmentations import *
from utils.transforms import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator


def save_profile_result(filename, table):
    import xlsxwriter
    workbook = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet()
    keys = ["Name", "Self CPU total %", "Self CPU total", "CPU total %" , "CPU total", \
            "CPU time avg", "Number of Calls"]
    for j in range(len(keys)):
        worksheet.write(0, j, keys[j])

    lines = table.split("\n")
    for i in range(3, len(lines)-4):
        words = lines[i].split(" ")
        j = 0
        for word in words:
            if not word == "":
                worksheet.write(i-2, j, word)
                j += 1
    workbook.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--num_iter", type=int, default=400, help="Total iteration of inference.")
    parser.add_argument("--num_warmup", type=int, default=10, help="Total iteration of warmup.")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    parser.add_argument('--channels_last', type=int, default=1, help='use channels last format')
    parser.add_argument('--arch', type=str, default="", help='model name')
    parser.add_argument('--profile', action='store_true', help='Trigger profile on current topology.')
    parser.add_argument('--precision', default='float32', help='Precision, "float32" or "bfloat16"')
    parser.add_argument('--ipex', action='store_true', help='Use IPEX.')
    parser.add_argument('--jit', action='store_true', help='Enable JIT.')
    parser.add_argument("--compile", action='store_true', default=False,
                    help="enable torch.compile")
    parser.add_argument("--backend", type=str, default='inductor',
                    help="enable torch.compile backend")
    parser.add_argument("--triton_cpu", action='store_true', default=False,
                    help="enable triton_cpu")

    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    if opt.triton_cpu:
        print("run with triton cpu backend")
        import torch._inductor.config
        torch._inductor.config.cpu_backend="triton"
    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    if opt.channels_last:
        oob_model = model
        oob_model = oob_model.to(memory_format=torch.channels_last)
        model = oob_model
        print("---- Use channels last format.")

    model.eval()  # Set in evaluation mode

    if opt.ipex:
        import intel_extension_for_pytorch as ipex
        print("Running with IPEX...")
        if opt.precision == "bfloat16":
            model = ipex.optimize(model, dtype=torch.bfloat16, inplace=True)
        else:
            model = ipex.optimize(model, dtype=torch.float32, inplace=True)
    else:
        model = model.to(device)

    dataloader = DataLoader(
        ImageFolder(opt.image_folder, transform= \
            transforms.Compose([DEFAULT_TRANSFORMS, Resize(opt.img_size)])),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()
    
    total_time = 0
    num_images = 0
    batch_time_list = []
    if opt.compile:
        model = torch.compile(model, backend=opt.backend, options={"freezing": True})

    if opt.precision == "bfloat16":
        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=True, dtype=torch.bfloat16):
            for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
                # Configure input
                input_imgs = Variable(input_imgs.type(Tensor))
                input_imgs = input_imgs.to(device)
                if opt.channels_last:
                    oob_inputs = input_imgs
                    oob_inputs = oob_inputs.contiguous(memory_format=torch.channels_last)
                    input_imgs = oob_inputs
                if opt.jit and batch_i == 0:
                    try:
                        model = torch.jit.trace(model, input_imgs, check_trace=False)
                        print("---- Use trace model.")
                    except:
                        model = torch.jit.script(model)
                        print("---- Use script model.")
                    if opt.ipex:
                        model = torch.jit.freeze(model)
                for i in range(opt.num_iter):
                    # Get detections
                    tic = time.time()
                    if opt.profile:
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as prof:
                            with torch.no_grad():
                                detections = model(input_imgs)
                                detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
                        #
                        if i == int(opt.num_iter/2):
                            import pathlib
                            timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
                            if not os.path.exists(timeline_dir):
                                os.makedirs(timeline_dir)
                            timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + '-' + \
                                        opt.arch + str(i) + '-' + str(os.getpid()) + '.json'
                            print(timeline_file)
                            prof.export_chrome_trace(timeline_file)
                            table_res = prof.key_averages().table(sort_by="cpu_time_total")
                            print(table_res)
                            # self.save_profile_result(timeline_dir + torch.backends.quantized.engine + "_result_average.xlsx", table_res)
                    else:
                        with torch.no_grad():
                            detections = model(input_imgs)
                            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
                    toc = time.time()
                    print("Iteration: {}, inference time: {} sec.".format(i, toc - tic), flush=True)
                    if i >= opt.num_warmup:
                        total_time += toc - tic
                        num_images += opt.batch_size
                        batch_time_list.append((toc - tic) * 1000)
                break
    elif opt.precision == "float16":
        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", enabled=True, dtype=torch.half):
            for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
                # Configure input
                input_imgs = Variable(input_imgs.type(Tensor))
                input_imgs = input_imgs.to(device)
                if opt.channels_last:
                    oob_inputs = input_imgs
                    oob_inputs = oob_inputs.contiguous(memory_format=torch.channels_last)
                    input_imgs = oob_inputs
                if opt.jit and batch_i == 0:
                    try:
                        model = torch.jit.trace(model, input_imgs, check_trace=False)
                        print("---- Use trace model.")
                    except:
                        model = torch.jit.script(model)
                        print("---- Use script model.")
                    if opt.ipex:
                        model = torch.jit.freeze(model)
                for i in range(opt.num_iter):
                    # Get detections
                    tic = time.time()
                    if opt.profile:
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as prof:
                            with torch.no_grad():
                                detections = model(input_imgs)
                                detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
                        #
                        if i == int(opt.num_iter/2):
                            import pathlib
                            timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
                            if not os.path.exists(timeline_dir):
                                os.makedirs(timeline_dir)
                            timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + '-' + \
                                        opt.arch + str(i) + '-' + str(os.getpid()) + '.json'
                            print(timeline_file)
                            prof.export_chrome_trace(timeline_file)
                            table_res = prof.key_averages().table(sort_by="cpu_time_total")
                            print(table_res)
                            # self.save_profile_result(timeline_dir + torch.backends.quantized.engine + "_result_average.xlsx", table_res)
                    else:
                        with torch.no_grad():
                            detections = model(input_imgs)
                            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
                    toc = time.time()
                    print("Iteration: {}, inference time: {} sec.".format(i, toc - tic), flush=True)
                    if i >= opt.num_warmup:
                        total_time += toc - tic
                        num_images += opt.batch_size
                        batch_time_list.append((toc - tic) * 1000)
                break
    else:
        for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
            # Configure input
            input_imgs = Variable(input_imgs.type(Tensor))
            input_imgs = input_imgs.to(device)
            if opt.channels_last:
                oob_inputs = input_imgs
                oob_inputs = oob_inputs.contiguous(memory_format=torch.channels_last)
                input_imgs = oob_inputs
            if opt.jit and batch_i == 0:
                try:
                    model = torch.jit.trace(model, input_imgs, check_trace=False)
                    print("---- Use trace model.")
                except:
                    model = torch.jit.script(model)
                    print("---- Use script model.")
                if opt.ipex:
                    model = torch.jit.freeze(model)
            for i in range(opt.num_iter):
                # Get detections
                tic = time.time()
                if opt.profile:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as prof:
                        with torch.no_grad():
                            detections = model(input_imgs)
                            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
                    #
                    if i == int(opt.num_iter/2):
                        import pathlib
                        timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
                        if not os.path.exists(timeline_dir):
                            os.makedirs(timeline_dir)
                        timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + '-' + \
                                    opt.arch + str(i) + '-' + str(os.getpid()) + '.json'
                        print(timeline_file)
                        prof.export_chrome_trace(timeline_file)
                        table_res = prof.key_averages().table(sort_by="cpu_time_total")
                        print(table_res)
                        # self.save_profile_result(timeline_dir + torch.backends.quantized.engine + "_result_average.xlsx", table_res)
                else:
                    with torch.no_grad():
                        detections = model(input_imgs)
                        detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

                toc = time.time()
                print("Iteration: {}, inference time: {} sec.".format(i, toc - tic), flush=True)
                if i >= opt.num_warmup:
                    total_time += toc - tic
                    num_images += opt.batch_size
                    batch_time_list.append((toc - tic) * 1000)
            break

    print("\n", "-"*20, "Summary", "-"*20)
    latency = total_time / num_images * 1000
    throughput = num_images / total_time
    print("inference latency:\t {:.3f} ms".format(latency))
    print("inference Throughput:\t {:.2f} samples/s".format(throughput))
    # P50
    batch_time_list.sort()
    p50_latency = batch_time_list[int(len(batch_time_list) * 0.50) - 1]
    p90_latency = batch_time_list[int(len(batch_time_list) * 0.90) - 1]
    p99_latency = batch_time_list[int(len(batch_time_list) * 0.99) - 1]
    print('Latency P50:\t %.3f ms\nLatency P90:\t %.3f ms\nLatency P99:\t %.3f ms\n'\
            % (p50_latency, p90_latency, p99_latency))

    '''
    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print("\nSaving images:")
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print("(%d) Image: '%s'" % (img_i, path))

        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                box_w = x2 - x1
                box_h = y2 - y1

                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(
                    x1,
                    y1,
                    s=classes[int(cls_pred)],
                    color="white",
                    verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                )

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = os.path.basename(path).split(".")[0]
        output_path = os.path.join("output", f"{filename}.png")
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0)
        plt.close()
        '''
