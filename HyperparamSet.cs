using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LSTM_Sharp
{
    /// <summary>
    /// Класс гиперпараметров модели LSTM
    /// </summary>
    public class HyperparamSet
    {
        /// <summary>
        /// Длина последовательности
        /// </summary>
        private int sequenceLength;

        /// <summary>
        /// Шаг смещения
        /// </summary>
        private int step;

        /// <summary>
        /// Число слоев
        /// </summary>
        private int layers;

        /// <summary>
        /// Число нейронов
        /// </summary>
        private int neurons;

        /// <summary>
        /// Скорость обучения
        /// </summary>
        private double learningRate;

        /// <summary>
        /// Количество эпох обучения
        /// </summary>
        private int epochs;

        /// <summary>
        /// Инициализация параметров по умолчанию
        /// </summary>
        internal HyperparamSet()
        {
            this.sequenceLength = 4;
            this.step = 1;
            this.layers = 1;
            this.neurons = 100;
            this.learningRate = 0.1;
            this.epochs = 50;
        }

        /// <summary>
        /// Конструктор для установки собственных значений гиперпараметров
        /// </summary>
        /// <param name="sequenceLength"> длина последовательности </param>
        /// <param name="step"> шаг </param>
        /// <param name="layers"> число слоев </param>
        /// <param name="neurons"> число нейронов </param>
        /// <param name="learningRate"> скорость обучения </param>
        /// <param name="epochs"> количество эпох </param>
        internal HyperparamSet(int sequenceLength, int step, int layers, int neurons, double learningRate, int epochs)
        {
            this.sequenceLength = sequenceLength;
            this.step = step;
            this.layers = layers;
            this.neurons = neurons;
            this.learningRate = learningRate;
            this.epochs = epochs;
        }

        /// <summary>
        /// Установка и получение значения длины последовательности
        /// </summary>
        public int SequenceLength 
        { 
            get { return sequenceLength; } 
            set { sequenceLength = value; } 
        }

        /// <summary>
        /// Установка и получение значения шага
        /// </summary>
        public int Step 
        {
            get { return step; }
            set { step = value; }
        }

        /// <summary>
        /// Установка и получение значения числа слоев
        /// </summary>
        public int Layers 
        {
            get { return layers; }
            set { layers = value; } 
        }

        /// <summary>
        /// Установка и получение значения числа нейронов
        /// </summary>
        public int Neurons 
        {
            get { return neurons; }
            set { neurons = value; }
        }

        /// <summary>
        /// Установка и получение значения скорости обучения
        /// </summary>
        public double LearningRate 
        {
            get { return learningRate; }
            set { learningRate = value; }
        }

        /// <summary>
        /// Установка и получение значения числа эпох
        /// </summary>
        public int Epochs 
        {
            get { return epochs; }
            set { epochs = value; }
        }
    }
}
