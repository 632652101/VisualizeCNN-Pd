if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torch
    from flashtorch.utils import apply_transforms, load_image

    image = load_image('content/images/king_penguin.jpg')

    from model.AlexNet_torch import alexnet

    model = alexnet(True)
    model.eval()

    from flashtorch.saliency.backprop import Backprop

    backprop = Backprop(model)

    owl = apply_transforms(image)
    out = model(owl)
    #
    target_class = 24
    # Ready to roll!
    visual_1, visual_2, images = backprop.visualize(owl, None, guided=False, use_gpu=False, return_output=True)
    visual_1 = visual_1.cpu().detach().numpy()
    visual_2 = visual_2.cpu().detach().numpy()
    #
    from reprod_log import ReprodLogger
    reprod_logger = ReprodLogger()
    reprod_logger.add("input", owl.cpu().detach().numpy())
    reprod_logger.add("out", out.cpu().detach().numpy())
    reprod_logger.add("visual_1", visual_1)
    reprod_logger.add("visual_2", visual_2)
    for idx, image in enumerate(images):
        if idx >= 3:
            break
        reprod_logger.add(f"images_{idx}", image.cpu().detach().numpy())
    reprod_logger.save("result_torch.npy")
