if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import paddle

    from model.AlexNet import alexnet

    model = alexnet(True)

    from saliency.ClassSpecificImageGenerationPP import PaddleVersion as ClassSpecific_pp

    class_specific_method = ClassSpecific_pp(model, target_class=130)

    out_list = class_specific_method.generate()

    # log
    from reprod_log import ReprodLogger
    reprod_logger = ReprodLogger()
    for idx, image in enumerate(out_list):
        reprod_logger.add(f"out_{idx}", image.cpu().detach().numpy())

    reprod_logger.save(f"npy/method_2/result_pp.npy")
