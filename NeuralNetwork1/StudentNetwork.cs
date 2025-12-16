using System;
using System.Diagnostics;
using System.Threading.Tasks;

namespace NeuralNetwork1
{
    public class Neuron
    {
        public int inputsCount;
        public double[] weights;
        public double output;

        public Neuron(int inputs)
        {
            inputsCount = inputs;
            weights = new double[inputsCount];
            // Random initialization of weights
            Random rand = new Random();
            for (int i = 0; i < inputsCount; i++)
            {
                weights[i] = rand.NextDouble();
            }
        }
        private double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        public double Compute(double[] input) 
        {
            double sum = 0.0;
            for (int i = 0; i < inputsCount; i++)
            {
                sum += input[i] * weights[i];
            }
            output = Sigmoid(sum);
            return output;
        }
    }
    public class Layer
    {
        public int inputsCount;
        public int neuronsCount;
        public Neuron[] neurons;
        public double[] output;

        public Layer(int neuronsCount, int inputsCount)
        {
            this.inputsCount = System.Math.Max(1, inputsCount);
            this.neuronsCount = System.Math.Max(1, neuronsCount);
            neurons = new Neuron[this.neuronsCount];
        }

        public double[] Compute(double[] input)
        {
            double[] array = new double[neuronsCount];
            for (int i = 0; i < neurons.Length; i++)
            {
                array[i] = neurons[i].Compute(input);
            }

            output = array;
            return array;
        }
    }

    public class StudentNetwork : BaseNetwork
    {
        public int inputsCount;
        public Layer[] layers;
        public int layersCount;
        public double[] output;
        public Stopwatch stopWatch = new Stopwatch();

        public double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        public StudentNetwork(int[] structure)
        {
            layersCount = structure.Length - 1;
            inputsCount = structure[0];

            layers = new Layer[layersCount];

            for (int i = 0; i < layersCount; i++)
            {
                int layerInputs  = structure[i];
                int layerNeurons = structure[i + 1];
                layers[i] = new Layer(layerNeurons, layerInputs)
                {
                    neurons = new Neuron[layerNeurons]
                };
                for (int j = 0; j < layerNeurons; j++)
                    layers[i].neurons[j] = new Neuron(layerInputs);
            }

            output = new double[structure[layersCount]];
        }

        private double[] Forward(double[] input, bool parallel)
        {
            double[] currentInput = input;

            for (int i = 0; i < layers.Length; i++)
            {
                Layer layer = layers[i];
                double[] layerInput = currentInput;   // вход в слой
                double[] layerOutput = new double[layer.neuronsCount];

                if (parallel)
                {
                    Parallel.For(0, layer.neuronsCount, j =>
                    {
                        Neuron neuron = layer.neurons[j];
                        layerOutput[j] = neuron.Compute(currentInput);
                        layer.output[j] = layerOutput[j];
                    });
                }
                else
                {
                    layer.Compute(currentInput);
                }

                currentInput = layerOutput;
            }

            return currentInput;
        }
        private void UpdateWeight(Layer layer, double[] input, int neuronIndex, double learningRate, double error)
        {
            Neuron neuron = layer.neurons[neuronIndex];
            for (int k = 0; k < neuron.inputsCount; k++)
            {
                neuron.weights[k] += learningRate * error * input[k];
            }
        }

        private void Backward(double[] input, double[] expectedOutput, double learningRate, bool parallel)
        {
            double[] currentInput = input;
            
            for (int i = layers.Length - 1; i >= 0; i--)
            {
                Layer layer = layers[i];
                double[] layerInput = currentInput;   // вход в слой
                double[] layerOutput = layer.output;   // выход из слоя
                if (i == layers.Length - 1)
                {
                    // Вычисление ошибки для выходного слоя
                    if (parallel)
                    {
                        Parallel.For(0, layer.neuronsCount, j =>
                        {
                            double error = expectedOutput[j] - layerOutput[j];
                            UpdateWeight(layer, layerInput, j, learningRate, error);
                        });
                    }
                    else
                    {
                        for (int j = 0; j < layer.neuronsCount; j++)
                        {
                            double error = expectedOutput[j] - layerOutput[j];
                            UpdateWeight(layer, layerInput, j, learningRate, error);
                        }
                    }
                }
                else
                {
                    // Вычисление ошибки для скрытых слоев (упрощённо)
                    if (parallel)
                    {
                        Parallel.For(0, layer.neuronsCount, j =>
                        {
                            double error = 0.0;
                            Layer nextLayer = layers[i + 1];
                            for (int k = 0; k < nextLayer.neuronsCount; k++)
                            {
                                error += nextLayer.neurons[k].weights[j] * (expectedOutput[k] - nextLayer.output[k]);
                            }
                            UpdateWeight(layer, layerInput, j, learningRate, error);
                        });
                    }
                    else
                    {
                        for (int j = 0; j < layer.neuronsCount; j++)
                        {
                            double error = 0.0;
                            Layer nextLayer = layers[i + 1];
                            for (int k = 0; k < nextLayer.neuronsCount; k++)
                            {
                                error += nextLayer.neurons[k].weights[j] * (expectedOutput[k] - nextLayer.output[k]);
                            }
                            UpdateWeight(layer, layerInput, j, learningRate, error);
                        }
                    }
                }
                currentInput = layer.output;
            }
        }

        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            int iterations = 0;
            double error = double.MaxValue;

            while (error > acceptableError)
            {
                double[] output = Forward(sample.input, parallel);
                Backward(sample.input, sample.Output, 0.1, parallel);
                double localError = 0.0;
                for (int i = 0; i < output.Length; i++)
                {
                    localError += Math.Pow(sample.Output[i] - output[i], 2);
                }
                error = Math.Min(error, localError);
                iterations++;
            }
            return iterations;
        }

        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            //  Сначала надо сконструировать массивы входов и выходов
            double[][] inputs = new double[samplesSet.Count][];
            double[][] outputs = new double[samplesSet.Count][];

            //  Теперь массивы из samplesSet группируем в inputs и outputs
            for (int i = 0; i < samplesSet.Count; ++i)
            {
                inputs[i] = samplesSet[i].input;
                outputs[i] = samplesSet[i].Output;
            }

            int epoch_to_run = 0;
            double error = double.MaxValue;

            stopWatch.Restart();

            while (epoch_to_run < epochsCount && error > acceptableError)
            {
                double localError = 0.0;
                for (int i = 0; i < samplesSet.Count; ++i)
                {
                    double[] output = Forward(inputs[i], parallel);
                    Backward(inputs[i], outputs[i], 0.1, parallel);
                    for (int j = 0; j < output.Length; j++)
                    {
                        localError += Math.Pow(outputs[i][j] - output[j], 2);
                    }
                }
                localError /= samplesSet.Count;
                error = Math.Min(error, localError);
                epoch_to_run++;


                OnTrainProgress((epoch_to_run * 1.0) / epochsCount, error, stopWatch.Elapsed);
            }

            OnTrainProgress(1.0, error, stopWatch.Elapsed);
            stopWatch.Stop();

            return error;
        }

        protected override double[] Compute(double[] input)
        {
            double[] array = input;
            for (int i = 0; i < layers.Length; i++)
            {
                array = layers[i].Compute(array);
            }

            output = array;
            return array;
        }
    }
}