from detectron2.config import CfgNode as CN


def add_yolos_config(cfg):
    """
    Add config for YOLOS.
    """
    cfg.MODEL.YOLOS = CN()
    cfg.set_new_allowed(True)
    # cfg.MODEL.YOLOS.DET_TOKEN_NUM = 
    # cfg.MODEL.YOLOS.BACKBONE_NAME = 
    # cfg.MODEL.YOLOS.INIT_PE_SIZE = 

    cfg.merge_from_file("../../configs/yolos_s_dwr.yaml")
