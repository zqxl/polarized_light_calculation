import numpy as np


def ema(Na, Nb, fa):
    """
    EMA计算(Effective medium approximation)
    :param Na: 介质a的复折射率 N = n-ik   N**2 = e(dielectric constant)
    :param Nb: 介质b的复折射率 N = n-ik
    :param fa: 介质a的体积占比
    :return: 等效复折射率  N = n-ik
    """
    ea = Na**2
    eb = Nb**2
    # 计算等效介电常数
    e = 1/4 * (-ea + 2*eb + 3*ea*fa - 3*eb*fa + np.sqrt(8*ea*eb + (-ea + 2*eb + 3*ea*fa - 3*eb*fa)**2))
    N = np.sqrt(e)
    return N
