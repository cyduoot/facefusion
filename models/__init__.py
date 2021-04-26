def create_model(opt):
    model = None
    if opt.model == 'pix2pix':
        assert(opt.dataset_mode == 'aligned')
        from .pix2pix_model import Pix2PixModel
        model = Pix2PixModel()
    elif opt.model == 'pix2pix_three':
        assert(opt.dataset_mode == 'aligned_three')
        from .pix2pix_model_three import Pix2PixModel
        model = Pix2PixModel()
    elif opt.model == 'pix2pix_attn':
        assert (opt.dataset_mode == 'aligned_three')
        from .pix2pix_model_attn import Pix2PixModel
        model = Pix2PixModel()
    elif opt.model == 'e2e':
        assert (opt.dataset_mode == 'paralleled')
        from .e2e_model import E2EModel
        model = E2EModel()
    else:
        raise NotImplementedError('model [%s] not implemented.' % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
