from functools import partial

import numpy as np
import torch
import time

from ddpg.ddpg import DDPG
from ddpg.utils import CosineAnnealingNormalNoise, ReplayBuffer
from utils.run_utils import evaluate_policy, fill_buffer_randomly, scale

# learn是optimization
# 无noise可得到最优模型
# 有noise
# act是
def learn(env_fn,
          epochs,
          buffer_size,
          eval_freq,
          eta_init,
          eta_min,
          reward_scale=5000,
          **algo_kwargs):

    # Build env and get related info
    env = env_fn()
    # LineEnv-v0
    obs_shp = env.observation_space.shape
    act_shp = env.action_space.shape
    act_dtype = env.action_space.dtype
    assert np.all(env.action_space.high == 1) & np.all(
        env.action_space.low == -1)

    # scalers
    # scale是归一化
    obs_scale = partial(scale, x_rng=env.observation_space.high)

    def rwd_scale(rew):
        return rew / reward_scale

    # explorative noise
    noise = CosineAnnealingNormalNoise(mu=np.zeros(act_shp),
                                       sigma=eta_init,
                                       sigma_min=eta_min,
                                       T_max=epochs)
    # model是optimization之后的，存下来，进行simulation的
    # 关于pytorch如何存model
    # construct model and buffer
    model = DDPG(obs_shp[0], act_shp[0], epochs=epochs, **algo_kwargs)

    obs, done = env.reset()


    buffer = ReplayBuffer(obs_shp[0],
                          act_shp[0],
                          maxlen=buffer_size,
                          act_dtype=act_dtype)
    
    # 这个程序里是把buffer存满了
    # 也可以 自己想把.....填满
    # 或者evaluate_policy
    # def evaluate_policy(env, policy, eval_epochs=10, scale_func=lambda x: x):
    # return avg_rwd, reward
    # policy  model
    evaluate_policy(env, model , eval_epochs=10, scale_func=lambda x: x)


    
    # fill_buffer_randomly(env_fn, buffer, obs_scale)
    # def fill_buffer_randomly(env_fn,buffer,scale_func=lambda x: x):
    # assert buffer.is_full()

    # init recorder
    q_opt = np.inf
    start = time.time()
    pop_opt = []
    try:
        # main loop
        for epoch in range(epochs):

         
            # obs, done = env.reset()
            # act = model.act(torch.as_tensor(obs, dtype=torch.float32))
            # print(len(act))
            # print((act))
     
          
            ret_ep, act_ep = 0., []
            obs, done = env.reset()
            print(len(obs))
            # 初始state 各个state假设一致
            # reset  return self.state, False
            #  self.max_svs = np.int64((self.t_end - self.t_start) * 3600 / headway_min) + 1
            # 145 N最大值
            for step in range(env.max_svs):
                # 这里是不是可以放存model
                act = model.act(torch.as_tensor(obs, dtype=torch.float32))
                # print(len(act))
                # print((act))

                act = np.clip(act + noise(), -1, 1)
                # clip这个函数将将数组中的元素限制在a_min, a_max之间，大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min。
                # print(act)
                # act 是从model里抽出来的
                # 没有model optimization环节的话，那就在这部分人工输入act
                # act是decision variable
                # env steps
                obs2, pwc, opc, done = env.step(act)
                # def step(self, action):return self.state, pwc, opc, done
                # act = (np.around(self.nopt * (action + 1) / 2) * self.var_int +
                # self.var_min).astype(np.int64)
                # print(len(act))
                # print(act)
                # 
                # print(obs2)


                rew = pwc + opc
                # total cost
                obs2 = obs_scale(obs2)

                # push experience into the buffer
                buffer.store(obs, act, rew, done, obs2)

                # record
                act_ep.append(act)
                ret_ep += rew

                obs = obs2

                # update the model
                # 根据epoch分隔得倒到局部最优和全局最优。这个可以用于画图。
            
                model.update(buffer, rwd_scale)

                if done:
                    # update the global optima
                    if q_opt >= ret_ep:
                        q_opt = ret_ep
                        pop_opt = np.asarray(act_ep)

                    # if epoch % 500 == 0:
                    #     print(f'EP: {epoch}|EpR: {ret_ep:.0f}| Q*: {q_opt:.0f}| T: {time.time()-start:.0f}|N:{noise.sigma:.3f}')
                        # epoch, local total cost, global total cost
                    print(f'EP: {epoch}|EpR: {ret_ep:.0f}| Q*: {q_opt:.0f}| T: {time.time()-start:.0f}|N:{noise.sigma:.3f}')
                       
                    noise.step()
                    model.lr_step()

                    break
    except KeyboardInterrupt:
        pass
    finally:
        print("done")
