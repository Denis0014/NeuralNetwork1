using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading.Tasks;

namespace NeuralNetwork1
{
    public class Neuron
    {
        public int inputsCount;
        public double[] weights;
        public double output;
        private static Random rand = new Random();

        public Neuron(int inputs)
        {
            inputsCount = inputs;
            weights = new double[inputsCount];
            // Random initialization of weights
            for (int i = 0; i < inputsCount; i++)
            {
                weights[i] = rand.NextDouble() * 0.25 - 0.125;
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

        private double[] Forward(double[] input)
        {
            double[] currentInput = input;

            for (int i = 0; i < layers.Length; i++)
            {
                Layer layer = layers[i];
                double[] layerOutput;

                layerOutput = layer.Compute(currentInput);

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

        // Исправлено: добавлен аргумент input для обновления первого слоя
        private void Backward(double[] input, double[] expectedOutput, double learningRate)
        {
            // Вычисление дельт для выходного слоя
            double[] deltaOut = new double[expectedOutput.Length];
            for (int k = 0; k < expectedOutput.Length; k++)
            {
                double outputValue = layers[layers.Length - 1].output[k];
                deltaOut[k] = outputValue * (1 - outputValue) * (expectedOutput[k] - outputValue);
            }

            // Вычисление дельт для скрытых слоев
            double[][] deltaHidden = new double[layersCount - 1][];
            double[] currentDelta = deltaOut;
            for (int k = layers.Length - 1; k >= 1; k--)
            {
                Layer currentLayer = layers[k];
                Layer previousLayer = layers[k - 1];
                deltaHidden[k - 1] = new double[previousLayer.neuronsCount];
                for (int j = 0; j < previousLayer.neuronsCount; j++)
                {
                    double sum = 0.0;
                    for (int m = 0; m < currentLayer.neuronsCount; m++)
                    {
                        sum += currentDelta[m] * currentLayer.neurons[m].weights[j];
                    }
                    double outputValue = previousLayer.output[j];
                    deltaHidden[k - 1][j] = outputValue * (1 - outputValue) * sum;
                }
                currentDelta = deltaHidden[k - 1];
            }

            // Обновление весов всех слоев
            // Исправлено: цикл теперь включает слой 0 (l >= 0)
            for (int l = layers.Length - 1; l >= 0; l--) {
                Layer layer = layers[l];
                // Исправлено: для нулевого слоя используем входной вектор сети
                double[] inputToUse = (l == 0) ? input : layers[l - 1].output;
                
                // Исправлено: правильный выбор массива дельт (deltaHidden[l] вместо deltaHidden[l-1])
                double[] currentDeltas = (l == layers.Length - 1) ? deltaOut : deltaHidden[l];

                for (int j = 0; j < layer.neuronsCount; j++)
                {
                    UpdateWeight(layer, inputToUse, j, learningRate, currentDeltas[j]);
                }
            }
        }

        private double[] ForwardParallel(double[] input)
        {
            double[] currentInput = input;
            for (int i = 0; i < layers.Length; i++)
            {
                Layer layer = layers[i];
                double[] layerInput = currentInput;   // вход в слой
                double[] layerOutput = new double[layer.neuronsCount];
                Parallel.For(0, layer.neuronsCount, j =>
                {
                    layerOutput[j] = layer.neurons[j].Compute(layerInput);
                });
                layer.output = layerOutput;
                currentInput = layerOutput;
            }
            return currentInput;
        }

        // Исправлено: добавлен аргумент input
        private void BackwardParallel(double[] input, double[] expectedOutput, double learningRate)
        {
            // Вычисление дельт для выходного слоя
            double[] deltaOut = new double[expectedOutput.Length];
            Parallel.For(0, expectedOutput.Length, k =>
            {
                double outputValue = layers[layers.Length - 1].output[k];
                deltaOut[k] = outputValue * (1 - outputValue) * (expectedOutput[k] - outputValue);
            });
            // Вычисление дельт для скрытых слоев
            double[][] deltaHidden = new double[layersCount - 1][];
            double[] currentDelta = deltaOut;
            for (int k = layers.Length - 1; k >= 1; k--)
            {
                Layer currentLayer = layers[k];
                Layer previousLayer = layers[k - 1];
                deltaHidden[k - 1] = new double[previousLayer.neuronsCount];
                Parallel.For(0, previousLayer.neuronsCount, j =>
                {
                    double sum = 0.0;
                    for (int m = 0; m < currentLayer.neuronsCount; m++)
                    {
                        sum += currentDelta[m] * currentLayer.neurons[m].weights[j];
                    }
                    double outputValue = previousLayer.output[j];
                    deltaHidden[k - 1][j] = outputValue * (1 - outputValue) * sum;
                });
                currentDelta = deltaHidden[k - 1];
            }
            // Обновление весов всех слоев
            for (int l = layers.Length - 1; l >= 0; l--)
            {
                Layer layer = layers[l];
                double[] inputToUse = (l == 0) ? input : layers[l - 1].output;
                double[] currentDeltas = (l == layers.Length - 1) ? deltaOut : deltaHidden[l];

                Parallel.For(0, layer.neuronsCount, j =>
                {
                    UpdateWeight(layer, inputToUse, j, learningRate, currentDeltas[j]);
                });
            }
        }

        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            int iterations = 0;
            double error = double.MaxValue;

            while (error > acceptableError)
            {
                double[] output;
                if (parallel)
                {
                    output = ForwardParallel(sample.input);
                    BackwardParallel(sample.input, sample.Output, 0.1);
                }
                else
                {
                    output = Forward(sample.input);
                    Backward(sample.input, sample.Output, 0.1);
                }
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
            double[][] inputs = new double[samplesSet.Count][];
            double[][] outputs = new double[samplesSet.Count][];

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
                    double[] output;
                    if (parallel)
                    {
                        output = ForwardParallel(inputs[i]);
                        BackwardParallel(inputs[i], outputs[i], 0.01);
                    }
                    else
                    {
                        output = Forward(inputs[i]);
                        Backward(inputs[i], outputs[i], 0.01);
                    }
                    for (int j = 0; j < output.Length; j++)
                    {
                        localError += Math.Pow(outputs[i][j] - output[j], 2);
                    }
                }
                error = localError;
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