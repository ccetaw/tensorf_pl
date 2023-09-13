from logger import logger

if __name__ == "__main__":
    import numpy as np
    test_image = (np.random.random(size=(800, 800, 3)) * 255).astype('uint8')
    test_dict = {'Name' : "Alice",
                 'Age' : 21,
                 'Degree' : "Bachelor Cse",
                 'University' : "Northeastern Univ"}

    logdir = 'log'
    logger.set_logdir(logdir)

    logger.debug_print(test_dict)
    logger.debug_print(test_image.shape)

    logger.write_dict2txt('test_dict.txt', test_dict)
    logger.write_dict2json('test_dict.json', test_dict)
    logger.write_image('imgs/test_image.png', test_image)

    logger.set_mode(debug=False)
    
    logger.debug_print(test_dict)
    logger.debug_print(test_image.shape)
    logger.write_image('imgs/test_image.png', test_image)


    for n_iter in range(10):
        logger.add_scalar('Loss/train', np.random.random(), n_iter)
        logger.add_scalar('Loss/test', np.random.random(), n_iter)
        logger.add_scalar('Accuracy/train', np.random.random(), n_iter)
        logger.add_scalar('Accuracy/test', np.random.random(), n_iter)
        logger.add_image('test', (np.random.random(size=(800, 800, 3)) * 255).astype('uint8'), n_iter, dataformats='HWC')






