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
        d, ri = sym.symbols('d%d, ri%d' % (layer_index, layer_index))  # 声明符号变量
        self.d = d
        self.ri = ri


class MultiLayerModel:
    """
    该类用来创建多层模型。使用方法如下：
    # 创建五层模型中的每一层（第0层为空气层）
    l1 = Layer(1, 1, 0)
    l2 = Layer(2, 1, 0)
    l3 = Layer(3, 1, 0)
    l4 = Layer(4, 1, 0)
    # 构建多层模型
    ml = MultiLayerModel([l1, l2, l3, l4])
    # 计算得到S矩阵的每一个元素S11，S22，S33，S44分别表示S矩阵四个元素。其中args应根据层数不同传入不同数量的参数。
    # 如下，其中wl为波长，ri(x)为第x层的复折射率，d(x)为第x层的厚度（单位：nm）。以下所有参数均为numpy.array，传入参数时应满足其广播机制。
    S11 = ml.S11(wl, ri0, d0, ri1, d1, ri2, d2, ri3, d3, ri4, d4)
    ... ...

    """
    def __init__(self, incident_angle=0, layers=None, wavelength=None):
        """
        多层模型
        :type wavelength: np.ndarray
        :param incident_angle: 入射角，角度
        :param layers: 由Layer对象为元素组成的列表,初始化时不包含空气层。从下标1-n分别代表第1-n层。
        :param wavelength: 波长,单位为nm
        """
        self.layers = layers or []
        # 插入空气层
        l_air = Layer(0, 0, 1)
        self.layers.insert(0, l_air)
        # 每一层的入射角
        self.incident_angles=[incident_angle]
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
        # 初始化S矩阵
        self.S = np.zeros([2, 2])

        self.get_S()

    def calculate_each_incident_angle(self):
        """
        计算每一层中光传播的角度的余弦值。使用的公式为《椭圆偏振测量术和偏振光》阿查姆P180,公式(4.3)
        :return:
        """
        for l in self.layers[1:]:
            cos_ia = sym.sqrt(1-(self.layers[0].ri/l.ri*sym.sin(self.incident_angles[0]))**2)
            self.incident_angles.append(cos_ia)

    @staticmethod
    def get_I(lp: Layer, lc: Layer):
        """
        就算当前层和下一层之间的界面矩阵 I
        :param lp: layer previous   上一层
        :param lc: layer current    当前层
        :return: 界面矩阵InterfaceMatrix。如果波长或者厚度之一为长度大于1的一维数组，则I就是三维矩阵，且第三维与前述一维数组对应
        """
        # 需要将p光和s光分开计算，待
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
        ib = 0 + 2*np.pi*l.d*l.ri/self.wl * sym.I
        L = np.array([[sym.exp(ib), 0], [0, sym.exp(-ib)]])

        return L

    def get_S(self):
        """
        计算S矩阵的表达式，其形式为由一系列符号变量组成的表达式
        :return:
        """
        # 计算所有层中的光线入射角
        self.calculate_each_incident_angle()
        # 计算I_01,即空气与第一层界面的界面矩阵
        scatter_matrix = self.get_I(self.layers[0], self.layers[1])
        # 循环计算S的表达式
        for i in range(1, len(self.layers)-1):
            scatter_matrix = np.dot(scatter_matrix, self.get_L(self.layers[i]))
            scatter_matrix = np.dot(scatter_matrix, self.get_I(self.layers[i], self.layers[i+1]))

        self.S = scatter_matrix
        # 将创建符号表达式转换成python函数，以便于利用numpy的广播Broadcasting机制进行高效的矩阵点乘运算
        all_symbol_variable = [self.wl]

        for l in self.layers:
            all_symbol_variable.extend([l.ri, l.d])

        self.S11 = sym.lambdify(all_symbol_variable, self.S[0, 0], "numpy")
        self.S12 = sym.lambdify(all_symbol_variable, self.S[0, 1], "numpy")
        self.S21 = sym.lambdify(all_symbol_variable, self.S[1, 0], "numpy")
        self.S22 = sym.lambdify(all_symbol_variable, self.S[1, 1], "numpy")

        print("多层模型S矩阵的表达式已经计算完成。调用MultiLayerModel.S11等函数时，请传入以下参数：")
        print(all_symbol_variable)




if __name__ == '__main__':
    # 创建五层模型中的每一层（第0层为空气层）
    l1 = Layer(1)
    l2 = Layer(2)
    l3 = Layer(3)
    l4 = Layer(4)
    # 构建多层模型
    ml = MultiLayerModel([l1, l2, l3, l4])

    pass
    pass

