# -*- coding: utf-8 -*
from configs import get_configs, set_global_seeds
from ddpg.learn import learn

from env import TubeEnv
import time


def main():
    settings = []

    for seed in range(1):
        # [0,1]
        stochastic = False
        settings.append((stochastic, 0.5, 0.5, seed))

    for setting in settings:
        stochastic, alpha, random_factor, seed = setting
 

        args, akwargs, ekwargs = get_configs(algo='ddpg',
                                             stochastic=stochastic,
                                             random_factor=random_factor,
                                             alpha=alpha,
                                             seed=seed)
                        
        set_global_seeds(args.seed)



        # 这两个是python中的可变参数。*args表示任何多个无名参数，它是一个tuple；**kwargs表示关键字参数，它是一个dict。
        # 并且同时使用*args和**kwargs时，必须*args参数列要在**kwargs前

        # simulation
        def env_fn():
        
            # print(TubeEnv(**ekwargs))
         
            return TubeEnv(**ekwargs)
            # LineEnv-v0
        
      

        # # simulation
        # def env_fn():

        #     return TubeEnv(**ekwargs)
 
        # optimization
        learn(env_fn, seed=args.seed, **akwargs)
  


if __name__ == '__main__':
    time_start=time.time()
    main()

    time_end=time.time()
    print('time cost',time_end-time_start,'s')

