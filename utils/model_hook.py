from utils.constant import DataType
import torch.nn as nn


class ModelHook(object):
    """ pytorch hook.
    """
    def __init__(self, model, blacklist=[],verbose=False):
        self.model = model
        self.blacklist = blacklist
        self.module_name = []
        self.layer_in = []
        self.layer_out = []
        self.verbose = verbose

    def hook(self, module, fea_in, fea_out):
        self.module_name.append(module.__class__)
        self.layer_in.append(fea_in)
        self.layer_out.append(fea_out.tolist()[0])
        return None

    def register_hook(self):
        for name, child in self.model.named_children():
            if isinstance(child, nn.Sequential):
                for cname, cchild in child.named_children():
                    if cname not in self.blacklist:
                        cchild.register_forward_hook(hook=self.hook)
                        if self.verbose:
                            print(f"layer:{cname} registered")
            else:
                if name not in self.blacklist:
                    if self.verbose:
                        print(f"layer:{name} registered")
                    child.register_forward_hook(hook=self.hook)

    def clear_cache(self):
        self.module_name = []
        self.layer_out = []
        self.layer_in = []


def get_instance_hooker(model_pyt, lnum):
    # Note that the blacklist varies according to the model. For the ACAS Xu model, we only need the
    # output of relu layer.
    blacklist = [f"h{i}" for i in range(lnum)]
    hooker = ModelHook(model_pyt, blacklist)
    hooker.register_hook()
    return hooker
