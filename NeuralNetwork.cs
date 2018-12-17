using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;

namespace NeuralNetwork
{
    [Serializable]
    public class NeuralNetwork
    {
        private double LearningRate { get; set; }
        private int InputVectorSize { get; set; }
        private int HidLayerSize { get; set; }
        private int OutLayerSize { get; set; }
        private List<Layer> Layers { get; set; }
        /// <summary>
        /// </summary>
        /// <param name="inputVectorSize">Размер входного слоя</param>
        /// <param name="hidLayerSize"> Размер скрытого слоя</param>
        /// <param name="outLayerSize"> Размер выходного слоя</param>
        /// <param name="learningRate"> Интенсивность обучения</param>
        public NeuralNetwork(int inputVectorSize, int hidLayerSize, int outLayerSize, double learningRate)
        {
            this.InputVectorSize = inputVectorSize;
            this.HidLayerSize = hidLayerSize;
            this.OutLayerSize = outLayerSize;
            this.LearningRate = learningRate;
        }

        /// <summary>
        /// Инициализирует сеть.
        /// </summary>
        public void InitNetwork()
        {
            Random rnd = new Random();
            Layers = new List<Layer>();

            Layer hiddLayer = new Layer(LayerTypes.HIDDEN);
            for (int a = 0; a < HidLayerSize; a++)
            {
                Neuron neuron = new Neuron();
                neuron.Output = 0.00;
                neuron.Inputs = new double[InputVectorSize];
                neuron.Weights = new double[InputVectorSize];
                for (int b = 0; b < InputVectorSize; b++)
                {
                    neuron.Inputs[b] = 0;
                    neuron.Weights[b] = 0.1 * rnd.NextDouble();
                    if (b % 2 == 0)
                    {
                        neuron.Weights[b] = -neuron.Weights[b];
                    }
                }
                neuron.Bias = 0.1 * rnd.NextDouble();
                hiddLayer.Neurons.Add(neuron);
            }
            Layers.Add(hiddLayer);

            Layer outLayer = new Layer(LayerTypes.OUTPUT);
            for (int a = 0; a < OutLayerSize; a++)
            {
                Neuron neuron = new Neuron();
                neuron.Output = 0.00;
                neuron.Inputs = new double[HidLayerSize];
                neuron.Weights = new double[HidLayerSize];
                for (int b = 0; b < HidLayerSize; b++)
                {
                    neuron.Inputs[b] = 0.00;
                    neuron.Weights[b] = 0.1 * rnd.NextDouble();
                    if (b % 2 == 0)
                    {
                        neuron.Weights[b] = -neuron.Weights[b];
                    }
                }
                neuron.Bias = 0.1 * rnd.NextDouble();
                outLayer.Neurons.Add(neuron);
            }
            Layers.Add(outLayer);
        }

        /// <summary>
        /// Инициализирует сеть данными из другой сети
        /// </summary>
        /// <param name="nn">Нейронная сеть</param>
        private void InitNetwork(NeuralNetwork nn)
        {
            this.Layers = nn.Layers;
            this.LearningRate = nn.LearningRate;
            this.OutLayerSize = nn.OutLayerSize;
            this.InputVectorSize = nn.InputVectorSize;
            this.OutLayerSize = nn.OutLayerSize;
        }

        /// <summary>
        /// Обучение сети
        /// </summary>
        /// <param name="data">Вектор входных значений</param>
        /// <param name="lbl">Истинное значение</param>
        /// <returns>Результат классификации</returns>
        public bool TrainNetwork(double[] data, int lbl)
        {
            double[] inputVector = GetInputVector(lbl);
            ForwardPropagate(data);
            BackPropagate(lbl);
            int classificated = GetClassification();
            return classificated == lbl;
        }

        /// <summary>
        /// Тестирование сети без обратного распространения ошибки
        /// </summary>
        /// <param name="data">вектор входных значений</param>
        /// <returns>классифицированное значение 0-9</returns>
        public int TestNetwork(double[] data)
        {
            ForwardPropagate(data);
            return GetClassification();
        }

        /// <summary>
        /// Метод обратного распространения ошибки
        /// </summary>
        /// <param name="target">Истинное значение</param>
        private void BackPropagate(int target)
        {
            Layer oLayer = Layers.Where(x => x.LayerType == LayerTypes.OUTPUT).First();
            for (int i = 0; i < oLayer.Neurons.Count; i++)
            {
                double output = oLayer.Neurons[i].Output;
                int targetOutput = i == target ? 1 : 0;
                double error = targetOutput - output;
                double weightsDelta = error * Sigmoiddx(output);
                UpdateWeights(oLayer, i, weightsDelta);
            }
            oLayer = Layers.Where(x => x.LayerType == LayerTypes.HIDDEN).First();
            for (int i = 0; i < oLayer.Neurons.Count; i++)
            {
                double output = oLayer.Neurons[i].Output;
                int targetOutput = i == target ? 1 : 0;
                double error = targetOutput - output;
                double weightsDelta = error * Sigmoiddx(output);
                UpdateWeights(oLayer, i, weightsDelta);
            }
        }
        /// <summary>
        /// Классификация
        /// </summary>
        /// <returns>Классифицированное значение</returns>
        private int GetClassification()
        {
            Layer oLayer = Layers.Where(x => x.LayerType == LayerTypes.OUTPUT).First();
            double max = 0.00;
            int maxIndex = 0;
            for (int i = 0; i < oLayer.Neurons.Count; i++)
            {
                if (oLayer.Neurons[i].Output > max)
                {
                    max = oLayer.Neurons[i].Output;
                    maxIndex = i;
                }
            }
            return maxIndex;
        }

        /// <summary>
        /// обновление весов
        /// </summary>
        /// <param name="layer">Слой</param>
        /// <param name="nodeId">Индекс нейрона</param>
        /// <param name="error">Ошибка</param>
        private void UpdateWeights(Layer layer, int nodeId, double error)
        {
            for (int j = 0; j < layer.Neurons[nodeId].Weights.Length; j++)
            {
                layer.Neurons[nodeId].Weights[j] += LearningRate * error * layer.Neurons[nodeId].Inputs[j];
            }
            layer.Neurons[nodeId].Bias += LearningRate * 1 * error;
        }

        /// <summary>
        /// Расчет выходных сигналов
        /// </summary>
        /// <param name="data">входной вектор</param>
        private void ForwardPropagate(double[] data)
        {
            TrainHiddenLayer(Layers[0], data);
            TrainOutputLayer(Layers[1]);
        }
        /// <summary>
        /// Активация скрытого слоя
        /// </summary>
        /// <param name="layer">слой</param>
        /// <param name="data">вектор входных значений</param>
        private void TrainHiddenLayer(Layer layer, double[] data)
        {
            for (int i = 0; i < layer.Neurons.Count; i++)
            {
                TrainNeuron(layer.Neurons[i], data);
            }
        }

        /// <summary>
        /// Активация нейрона
        /// </summary>
        /// <param name="neuron">Нейрон</param>
        /// <param name="data">Вектор входных значений</param>
        private void TrainNeuron(Neuron neuron, double[] data)
        {
            neuron.Output = neuron.Bias;
            for (int i = 0; i < neuron.Inputs.Length; i++)
            {
                neuron.Inputs[i] = data[i];
                neuron.Output += neuron.Inputs[i] * neuron.Weights[i];
            }
            neuron.Output = Sigmoid(neuron.Output);
        }

        /// <summary>
        /// Активация выходного слоя
        /// </summary>
        /// <param name="layer">слой</param>
        private void TrainOutputLayer(Layer layer)
        {
            double[] input = GetOutputVector(Layers[0]);
            for (int i = 0; i < layer.Neurons.Count; i++)
            {
                TrainNeuron(layer.Neurons[i], input);
            }
        }

        /// <summary>
        /// Получение вектора по цифре. Пример: 3 = [0,0,0,1,0,0,0,0,0,0]
        /// </summary>
        /// <param name="lbl">Цифра</param>
        /// <returns></returns>
        private double[] GetInputVector(int lbl)
        {
            double[] input = new double[10];
            for (int i = 0; i < 10; i++)
            {
                input[i] = lbl == i ? 1 : 0;
            }
            return input;
        }
        /// <summary>
        /// Получение вектора значений выходных сигналов слоя
        /// </summary>
        /// <param name="l">Слой</param>
        /// <returns>Вектор значений</returns>
        private double[] GetOutputVector(Layer l)
        {
            double[] vector = new double[l.Neurons.Count];
            for (int i = 0; i < l.Neurons.Count; i++)
            {
                vector[i] = l.Neurons[i].Output;
            }
            return vector;
        }

        /// <summary>
        /// Сохранить текущее состояние сети.
        /// </summary>
        /// <param name="path">Путь к файлу формат .dat</param>
        public void SaveNetwork(string path)
        {
            BinaryFormatter formatter = new BinaryFormatter();
            using (FileStream fs = new FileStream(path, FileMode.OpenOrCreate))
            {
                formatter.Serialize(fs, this);
            }
        }
        /// <summary>
        /// Загрузить сеть из файла.
        /// </summary>
        /// <param name="path">Путь к файлу .dat</param>
        public void LoadNetwork(string path)
        {
            BinaryFormatter formatter = new BinaryFormatter();
            NeuralNetwork nn;
            using (FileStream fs = new FileStream(path, FileMode.Open))
            {
                nn = formatter.Deserialize(fs) as NeuralNetwork;
            }
            InitNetwork(nn);
        }

        private double Sigmoid(double value) => 1.00 / (1.00 + Math.Exp(-value));
        private double Sigmoiddx(double value) => value * (1 - value);
    }
}
