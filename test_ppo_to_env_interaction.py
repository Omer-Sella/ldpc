

def test_logicalXOnAction():
    import gymnasium as gym
    import qecc
    from qecc.bb_gym import exampleDecoderFunction
    import torch
    
    
    def environmentFunction():
        return gym.make('qecc/bbcode-v0', l = 6, m = 6, evaluationDecoderFunction = exampleDecoderFunction, errorRange = [0.01, 0.001])
    
    env = environmentFunction()
    env.reset()
    action = torch.tensor([0., 1., 0., 1., 1., 0., 1., 0., 0., 1., 1., 0., 0., 0., 1., 1., 1., 1.,
        0., 1., 0., 1., 0., 1., 1., 0., 1., 1., 1., 0., 1., 1., 0., 1., 0., 0.,
        1., 1., 1., 0., 1., 0., 1., 0., 0., 1., 1., 1., 0., 1., 0., 1., 0., 0.,
        1., 0., 1., 1., 1., 1., 0., 0., 0., 1., 0., 0., 1., 1., 0., 1., 0., 0.,
        1., 1., 0., 1., 1., 0., 0., 0., 1., 1., 0., 1., 1., 0., 0., 1., 0., 1.,
        1., 1., 0., 0., 0., 1., 0., 1., 0., 1., 1., 0., 0., 0., 0., 1., 1., 1.,
        1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0.,
        1., 0., 0., 1., 0., 1., 1., 0., 1., 0., 0., 1., 0., 1., 0., 1., 1., 1.])
    action = action.numpy()
    next_o, r, d, _, info = env.step(action)

if __name__ == '__main__':
    test_logicalXOnAction()