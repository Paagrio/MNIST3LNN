using System;
using System.Collections.Generic;

namespace NeuralNetwork
{
    [Serializable]
    sealed class Layer
    {
        public LayerTypes LayerType { get; set; }
        public List<Neuron> Neurons { get; set; }

        public Layer(LayerTypes layerType)
        {
            LayerType = layerType;
            Neurons = new List<Neuron>();
        }
    }
}