import argparse
from config import get_parameters
from train import train
from test import test
#from analysis import assess

def main(config):
            
    if config['mode'] == 'train':

        train(config)

    elif config['mode'] == 'test':

        test(config)

    elif config['mode'] == 'assess':

        pass #assess(config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Modelo de transferencia de aprendizaje para\
                                                  segmentacion semantica de displasia cortical\
                                                  focal en imagenes de resonancia magnetica')
    parser.add_argument('-m', 
                        '--mode', 
                        choices=['train', 'test', 'assess'],
                        default='train',
                        help="Modo de operacion del programa.\n \
                        Las opciones son: \
                        'train', 'test' y 'assess'\n \
                        La opcion por defecto es 'train'"
    )
    args = parser.parse_args()
    config = get_parameters(args.mode)
    main(config)

