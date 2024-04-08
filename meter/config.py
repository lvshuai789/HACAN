from sacred import Experiment

ex = Experiment("METER")


def _loss_names(d):
    ret = {
        "itm": 0,
        "mlm": 0,
        "mpp": 0,
        "vqa": 0,
        "vcr": 0,
        "vcr_qar": 0,
        "nlvr2": 0,
        "irtr": 0,
        "contras": 0,
        "snli": 0,
    }
    ret.update(d)
    return ret


@ex.config
def config():
    exp_name = "finetune_irtr_f30k"
    seed = 0
    datasets = "f30k"
    loss_names = _loss_names({"irtr": 1})
    batch_size = 64  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.
    margin = 0.2

    # Image setting
    #train_transform_keys = ["clip"]
    #val_transform_keys = ["clip"]
    image_size = 224
    patch_size = 32
    #draw_false_image = 1
    image_only = False

    # Text Setting
    #vqav2_label_size = 3129
    max_text_len = 32
    tokenizer = "/home/ls/bert-base-uncased"
    vocab_size = 30522
    whole_word_masking = False # note that whole_word_masking does not work for RoBERTa
    mlm_prob = 0.15
    #draw_false_text = 0

    # Transformer Setting
    num_top_layer = 6
    input_image_embed_size = 1024
    input_text_embed_size = 768
    vit = "swin_base_patch4_window7_224_in22k"
    hidden_size = 768
    num_heads = 12
    num_layers = 6
    mlp_ratio = 4
    drop_rate = 0.1

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 1e-4
    lr_update = 10
    weight_decay = 0.01
    decay_power = 1
    max_epoch = 30
    max_steps = None
    warmup_steps = 10000
    end_lr = 0
    lr_mult_head = 5  # multiply lr for downstream heads
    lr_mult_cross_modal = 5  # multiply lr for the cross-modal module

    # Downstream Setting
    get_recall_metric = False

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False
    checkpoint = '/data3/lihaoxuan/New_Time/TKDE/github/runs/i2t_freeze/epoch=68-step=172499-v1.ckpt'

    # below params varies with the environment
    data_root = '/data1/lihaoxuan/orignal-datasets/'
    log_dir = "result"
    per_gpu_batchsize = 64  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 1
    num_nodes = 1
    load_path = ""
    num_workers = 8
    precision = 16

    #SCAN
    direction = 't2i'
    lambda_softmax = 9
    embed_dim = 768

    # add
    margin = 0.2
    loss = "GCD"  # F_HN / M_HN / GCD
    activation = 'leaky_relu'  # tanh / relu / leaky_relu
    experiment_name = ''
    fold5 = False
    save_path=''
    # save_path='runs/i2t_freeze_f30k_last_but_one_t2i'
    focal_type = 'prob'



@ex.named_config
def coco_config():
    exp_name = "finetune_irtr_coco"
    seed = 0
    datasets = "coco"
    loss_names = _loss_names({"irtr": 1})
    batch_size = 64  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.
    margin = 0.2

    # Image setting
    #train_transform_keys = ["clip"]
    #val_transform_keys = ["clip"]
    image_size = 224
    patch_size = 32
    #draw_false_image = 1
    image_only = False

    # Text Setting
    #vqav2_label_size = 3129
    max_text_len = 32
    tokenizer = "/home/ls/bert-base-uncased"
    vocab_size = 30522
    whole_word_masking = False # note that whole_word_masking does not work for RoBERTa
    mlm_prob = 0.15
    #draw_false_text = 0

    # Transformer Setting
    num_top_layer = 6
    input_image_embed_size = 1024
    input_text_embed_size = 768
    vit = "swin_base_patch4_window7_224_in22k"
    hidden_size = 768
    num_heads = 12
    num_layers = 6
    mlp_ratio = 4
    drop_rate = 0.1

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 1e-4
    lr_update = 10
    weight_decay = 0.01
    decay_power = 1
    max_epoch = 30
    max_steps = None
    warmup_steps = 10000
    end_lr = 0
    lr_mult_head = 5  # multiply lr for downstream heads
    lr_mult_cross_modal = 5  # multiply lr for the cross-modal module

    # Downstream Setting
    get_recall_metric = False

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False
    checkpoint = '/data3/lihaoxuan/New_Time/TKDE/github/runs/i2t_freeze/last.ckpt'

    # below params varies with the environment
    data_root = '/data1/lihaoxuan/orignal-datasets/'
    log_dir = "result"
    per_gpu_batchsize = 64  # you should define this manually with per_gpu_batch_size=#
    num_gpus = 1
    num_nodes = 1
    load_path = ""
    num_workers = 8
    precision = 16

    #SCAN
    direction = 'i2t'
    lambda_softmax = 9

    # add
    margin = 0.2
    loss = "GCD"
    activation = 'leaky_relu'
    experiment_name = ''
    fold5 = False
    save_path='runs/i2t_freeze_coco_last_layers'
    focal_type = 'prob'

