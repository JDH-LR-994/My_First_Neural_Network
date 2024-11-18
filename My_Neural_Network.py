import numpy
import scipy.special  # необходимо для сигмоиды
import matplotlib.pyplot
import class_neuralNetwork



# Задаём кол-во входных, скрытых и выходных узлов
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# коэффициент обучения
learning_rate = 0.1

# создаём экземпляр класса нейронки
n = class_neuralNetwork.neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# открытие и считывание тренировочной выборки
training_data_file = open("mnist_dataset/mnist_train_100.csv", "r")
training_data_list = training_data_file.readlines()
training_data_file.close()

# Тренеровка сети
epochs = 5
for e in range(epochs):
    # перебрать все записи в тренировочном наборе данных
    for record in training_data_list:
        # получить список значений, используя символ (',')
        # в качестве разделителя
        all_values = record.split(',')
        # масштабировать и сместить входные значения
        inputs = (numpy.asarray(all_values[1:]).astype('float') / 255.0 * 0.99) + 0.01
        # создание целевые значения выходных значения (все равны 0.01, за исключением
        # желаемого маркерного значения, равного 0.99
        targets = numpy.zeros(output_nodes) + 0.01

        # all_values[0] - целевое значение для данной записи
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

# загрузить в список тестовый набор данных CSV-файла набора MNIST
test_data_file = open("mnist_dataset/mnist_test_10.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()

# Тестирование нейронной сети

# журнал оценок работы ии, первоначально пустой
scorecord = []

# Перебрать все записи в тестовом наборе данных
for record in test_data_list:
    # получить список значений, используя символ (',')
    # в качестве разделителя
    all_values = record.split(',')
    # Правильное значение
    correct_label = all_values[0]
    print("Ожидаемый ответ сети:", correct_label)
    inputs = (numpy.asarray(all_values[1:]).astype('float') / 255.0 * 0.99) + 0.01
    image_number = (numpy.asarray(all_values[1:]).astype('float')).reshape((28, 28))
    matplotlib.pyplot.imshow(image_number, cmap = "Greys", interpolation=None)
    matplotlib.pyplot.show()
    # опрос сети
    outputs = n.query(inputs)
    # индекс наибольшего значения, прогноз иишки
    label = numpy.argmax(outputs)
    print("Ответ сети:", label)

    print()
    if int(label) == int(correct_label):
        scorecord.append(1)
    else:
        scorecord.append(0)

#рассчёт показателя эффективности в виде
#доли правильных ответов

scorecord_array = numpy.asarray(scorecord)
print("Эффективность: ", (scorecord_array.sum()/scorecord_array.size) * 100 , "процентов")