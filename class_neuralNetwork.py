import numpy
import scipy.special  # необходимо для сигмоиды


class neuralNetwork:
    # инициализация нейронки
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # Передаём кол-во входных, скрытых и выходных узлов
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # коэффициент обучения
        self.lr = learningrate

        # Матрицы весовых коэффициентов связей wih (вход + скрытые)
        # и who (скрытые + выходные)
        # весовые коэффициенты связей между узлом i и узлом j следующего слоя
        # обозначены как w_i_j
        # w11 w12
        # w21 w22 итд
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # использование сигмоиды (1/(1 + e ^ x)) в качестве функции активации
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    # тренировка нейронной сети
    def train(self, inputs_list, target_list):
        # преобразуем список входных значений в двумерный массив
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(target_list, ndmin=2).T

        # расчёт входящих сигналов для скрытого слоя
        hidden_inputs = numpy.dot(self.wih, inputs)
        # расчёт выходящих сигналов из скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)

        # рассчитать входящие сигналы для выходного слоя
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # рассчитать исходящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)

        # ошибка = целевое значение - фактическое значение
        output_errors = targets - final_outputs

        # ошибки скрытого слоя - это ошибки output_errors,
        # распределённые пропорционально весовым коэффициентам связей
        # и рекомбинированные на скрытых узлах
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # обновить весовые коэффициенты для связей
        # между скрытым и выходным слоями

        self.who += self.lr * numpy.dot(
            (output_errors * final_outputs * (1.0 - final_outputs)),
            numpy.transpose(hidden_outputs)
        )


        # обновить весовые коэффициенты для связей
        # между входными и скрытыми слоями
        self.wih += self.lr * numpy.dot(
            (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
            numpy.transpose(inputs)
        )

    # опрос нейронки
    def query(self, inputs_list):
        # преобразуем список входных значений
        # в двумерный массив
        inputs = numpy.array(inputs_list, ndmin=2).T

        # расчёт входящих сигналов для скрытого слоя
        hidden_inputs = numpy.dot(self.wih, inputs)
        # расчёт выходящих сигналов из скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)

        # рассчитать входящие сигналы для выходного слоя
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # рассчитать исходящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
