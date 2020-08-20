import re
from collections import defaultdict

import torch
import matplotlib.pyplot as plt
from matplotlib import patches


def xyxy2xywh(box):
    x1, y1, x2, y2 = box.T
    new_box = torch.stack((x1, y1, x2 - x1, y2 - y1), dim=1)
    return new_box


colors = (
    (0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 0, 0),(0, 1, 0), 
    (0.8, 1, 1), (0.5, 0.4, 0.7), (1, 0.6, 0), (1, 0.4, 0.6), (0.4, 0.6, 0.4), 
)

def factor(x):
    # RGB color factors
    if x < 10:
        return colors[x]
    
    i, n = x % 5, x // 5
    base = 0.7 ** n
    extra = 1 - base
    if i < 3:
        f = [base, base, base]
        f[i] = extra
    else:
        f = [extra, extra, extra]
        f[i - 3] = base
    return f


def show(images, targets=None, classes=None, save=""):
    if isinstance(images, torch.Tensor) and images.dim() == 3:
        images = [images]
    if isinstance(targets, dict):
        targets = [targets]
    if isinstance(save, str):
        save = [save] * len(images)
        
    for i in range(len(images)):
        show_single(images[i], targets[i] if targets else targets, classes, save[i])

    
def show_single(image, target, classes, save):
    """
    Show the image, with or without the target
    Arguments:
        image (tensor[3, H, W]): RGB channels, value range: [0.0, 1.0]
        target (dict[tensor]): current support "boxes", "labels", "scores", "masks"
           all tensors should be of the same length, assuming N
           masks: shape=[N, H, W], dtype=torch.float
        classes (tuple): class names
        save (str): path where to save the figure
    """
    image = image.clone()
    if target and "masks" in target:
        masks = target["masks"].unsqueeze(1)
        masks = masks.repeat(1, 3, 1, 1)
        for i, m in enumerate(masks):
            f = torch.tensor(factor(i)).reshape(3, 1, 1).to(image)
            value = f * m
            image += value
            
    image = image.clamp(0, 1)
    H, W = image.shape[-2:]
    fig = plt.figure(figsize=(W / 72, H / 72))
    ax = fig.add_subplot(111)
    
    im = image.cpu().numpy()
    ax.imshow(im.transpose(1, 2, 0)) # RGB
    ax.set_title("W: {}   H: {}".format(W, H))
    ax.axis("off")

    if target:
        if "labels" in target:
            if classes is None:
                raise ValueError("'classes' should not be None when 'target' has 'labels'!")
            tags = {l: i for i, l in enumerate(tuple(set(target["labels"].tolist())))}
            
        index = 0
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = xyxy2xywh(boxes).cpu().detach()
            for i, b in enumerate(boxes):
                if "labels" in target:
                    l = target["labels"][i].item()
                    index = tags[l]
                    txt = classes[l]
                    if "scores" in target:
                        s = target["scores"][i]
                        s = round(s.item() * 100)
                        txt = "{} {}%".format(txt, s)
                    ax.text(
                        b[0], b[1], txt, fontsize=10, color=factor(index),  
                        horizontalalignment="left", verticalalignment="bottom",
                        bbox=dict(boxstyle="square", fc="black", lw=1, alpha=1)
                    )
                    
                    
                rect = patches.Rectangle(b[:2], b[2], b[3], linewidth=2, edgecolor=factor(index), facecolor="none")
                ax.add_patch(rect)

    if save:
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(save)
    plt.show()
    
    
def parse(txt):
    result = {}
    result["epoch"] = int(re.search(r"^epoch: (\d+)", txt, re.MULTILINE).group(1))
    result["lr_epoch"] = float(re.search(r"lr_epoch: (\d+.\d+)", txt).group(1))
    result["factor"] = float(re.search(r"factor: (\d+.\d+)", txt).group(1))
    
    outputs = re.findall(r"^(\d+)\s+(\d+.\d+)\s+(\d+.\d+)\s+(\d+.\d+)\n", txt, re.MULTILINE)
    iters, loss_box, loss_obj, loss_cls = zip(*outputs)
    iters = [int(x) for x in iters]
    loss_box = [float(x) for x in loss_box]
    loss_obj = [float(x) for x in loss_obj]
    loss_cls = [float(x) for x in loss_cls]
    
    result["iters"] = iters[len(iters) // 2]
    result["loss_box"] = sum(loss_box) / len(loss_box)
    result["loss_obj"] = sum(loss_obj) / len(loss_obj)
    result["loss_cls"] = sum(loss_cls) / len(loss_cls)
    
    result["bbox_AP"] = float(re.search(r"'bbox AP': (\d+.\d+)", txt).group(1))
    return result


def gather(captures):
    results = [parse(txt) for txt in captures]
    results.sort(key=lambda x: x["epoch"])
    
    info = defaultdict(list)
    for res in results:
        info["epoch"].append(res["epoch"])
        info["lr_epoch"].append(res["lr_epoch"])
        info["factor"].append(res["factor"])
        info["bbox_AP"].append(res["bbox_AP"])
        
        info["iters"].append(res["iters"])
        info["loss_box"].append(res["loss_box"])
        info["loss_obj"].append(res["loss_obj"])
        info["loss_cls"].append(res["loss_cls"])
    return info
        
    
def plot(paths, x="epoch", y=["bbox_AP"], length=None, legend=True):
    if isinstance(paths, str):
        paths = [[paths]]
    else:
        for i, path in enumerate(paths):
            if isinstance(path, str):
                paths[i] = [path]
                
    if isinstance(y, str):
        y = [y]
    if isinstance(length, int):
        length = [length] * len(paths)
        
    for i, path in enumerate(paths):
        captures = []
        for p in path:
            with open(p, "r") as f:
                text = f.read()
            captures.extend(re.findall(r"(epoch: \d+\n.*?)}\n", text, re.DOTALL))
        if length is not None:
            captures = captures[:length[i]]
            
        data = gather(captures)

        x_axis = data[x]
        for k in y:
            label = "{} {}".format(i, k) if len(paths) > 1 else "{}".format(k)
            plt.plot(x_axis, data[k], label=label)
    if legend:
        plt.legend()
    plt.xlabel(x)
    
