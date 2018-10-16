import numpy as np
import sympy as sym


class Layer:
    def __init__(self, layer_index, thickness=0, refraction_indices=0):
        """
        创建一层
        :param layer_index: 层数编号。第0层始终默认为空气，创建时请从第1层开始，自上向下逐渐递增
        :param thickness:  厚度,单位为nm
        :param refraction_indices: 复折射率 N=n-ik
        """

        self.thickness = thickness
        self.refraction_indices = refraction_indices

        # 次数使用了符号变量，用来推导算式。推导结束后将使用实际的数据代替相应的符号
        d, ri = sym.symbols('d%d, ri%d' % (layer_index, layer_index))  # 声明符号变量，否则会认为没有定义
        self.d = d
        self.ri = ri


class MultiLayerModel:
    def __init__(self, layers=None, wavelength=None):
        """
        多层模型
        :type wavelength: np.ndarray
        :param layers: 由Layer对象为元素组成的列表,初始化时不包含空气层。从下标1-n分别代表第1-n层。
        :param wavelength: 波长,单位为nm
        """
        self.layers = layers or []
        # 实际的波长数值，为一维数组，
        self.wavelength = wavelength or np.array([0])
        # 符号变量用来表示波长
        wl = sym.symbols('wl')
        self.wl = wl
        # S矩阵四个位置上的数值，四个变量为一维数组，长度1-n取决于输入的波长范围和层厚范围
        self.S11 = 0
        self.S12 = 0
        self.S21 = 0
        self.S22 = 0

    @staticmethod
    def get_I(lp: Layer, lc: Layer):
        """
        就算当前层和下一层之间的界面矩阵 I
        :param lp: layer previous   上一层
        :param lc: layer current    当前层
        :return: 界面矩阵InterfaceMatrix。如果波长或者厚度之一为长度大于1的一维数组，则I就是三维矩阵，且第三维与前述一维数组对应
        """
        # 透射系数
        t = 2 * lp.ri * (lp.ri+lc.ri)
        # 反射系数
        r = (lp.ri-lc.ri) / (lp.ri+lc.ri)
        # 计算界面矩阵
        M = np.array([[1, r], [r, 1]])
        InterfaceMatrix = 1/t*M
        return InterfaceMatrix

    def get_L(self, l: Layer):
        """
        计算l层的特征矩阵
        :param l: 层对象
        :return: l层的特征矩阵
        """
        # 待解决，此处可能要手动计算exp(b)
        # 参考https://stackoverflow.com/questions/27190750/sympy-typeerror-cant-convert-expression-to-float
        # ib = sym.complex(0, 2*np.pi*l.d*l.ri/self.wl)
        ib = 0 + 2*np.pi*l.d*l.ri/self.wl * sym.I
        L = np.array([[sym.exp(ib), 0], [0, sym.exp(-ib)]])

        return L

    def get_S(self):
        """
        计算S矩阵的表达式，其形式为由一系列符号变量组成的表达式
        :return:
        """
        # 插入空气层
        l_air = Layer(0, 0, 1)
        self.layers.insert(0, l_air)
        # 计算I_01,即空气与第一层界面的界面矩阵
        S = self.get_I(self.layers[0], self.layers[1])
        # 循环计算S的表达式
        for i in range(1, len(self.layers)-1):
            S = np.dot(S, self.get_L(self.layers[i]))
            S = np.dot(S, self.get_I(self.layers[i], self.layers[i+1]))
        return S

    def caculate_S(self):
        """
        将一系列数据带入get_S()得出的表达式
        :return:
        """


if __name__ == '__main__':
    l1 = Layer(1, 1, 0,)
    l2 = Layer(2, 1, 0,)
    l3 = Layer(3, 1, 0,)

    ml = MultiLayerModel([l1, l2, l3])
    I = ml.get_I(l1, l2)
    L = ml.get_L(l1)

    S = ml.calculate_S()
    pass

