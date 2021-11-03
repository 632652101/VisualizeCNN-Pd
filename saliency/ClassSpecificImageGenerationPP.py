import os

import numpy as np
import paddle.optimizer

from .utils.misc_functions_pp import (preprocess_image,
                                      recreate_image,
                                      save_image)


class PaddleVersion:

    def __init__(self, model, target_class=1, total_classes=1000):
        self.model = model
        self.model.eval()
        self.target_class = target_class
        np.random.seed(10)
        self.created_image = np.uint8(np.random.uniform(0, 255, (224, 224, 3)))
        if not os.path.exists('images/method_2/class_' + str(self.target_class) + '/pp'):
            os.makedirs('images/method_2/class_' + str(self.target_class) + '/pp')

    def generate(self,
                 iterations=160):
        """Generates class specific image
            Keyword Arguments:
                iterations {int} -- Total iterations for gradient ascent (default: {150})
            Returns:
                np.ndarray -- Final maximally activated class image
        """
        init_lr = 6.
        out_list = []
        for i in range(1, iterations):
            self.processed_image = preprocess_image(self.created_image, False)

            # Define optimizer
            optimizer = paddle.optimizer.SGD(learning_rate=init_lr,
                                             parameters=[self.processed_image],
                                             weight_decay=0.02)


            # Forward
            out = self.model(self.processed_image)
            out_list.append(out)

            # Target specific loss
            loss = -out[0][self.target_class]

            if i % 10 == 0 or i == iterations - 1:
                print('Iteration:', str(i), 'Loss',
                      "{0:.2f}".format(loss.cpu().detach().numpy()[0]))

            # Zero grads
            optimizer.clear_grad()
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(self.processed_image)

            if i % 10 == 0 or i == iterations - 1:
                # save image
                im_path = 'images/method_2/class_' + str(self.target_class) + '/pp/c_' + str(
                    self.target_class) + '_' + 'iter_' + str(i) + '_pp.png'
                save_image(self.created_image, im_path)

        return out_list
