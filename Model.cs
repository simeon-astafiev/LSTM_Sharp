using System;
using System.Collections.Generic;
using System.Data;
using System.Globalization;
using System.IO;
using System.Linq;
using Keras.Layers;
using Keras.Models;
using Numpy;
using Numpy.Models;

namespace LSTM_Sharp
{
    /// <summary>
    /// Класс взаимодействия с моделью машинного обучения
    /// </summary>
    public class Model
    {
        /// <summary>
        /// Экземпляр модели
        /// </summary>
        private Sequential model = new Sequential();

        /// <summary>
        /// Набор гиперпараметров по умолчанию
        /// </summary>
        private HyperparamSet set = new HyperparamSet();

        /// <summary>
        /// Флаг оконного метода: если true, используем его
        /// </summary>
        private bool isWindowing = true;

        /// <summary>
        /// Размер пакета
        /// </summary>
        private int batchSize = 32;

        /// <summary>
        /// Столбцы по которым будем строить таблицу признаков 
        /// </summary>
        private string[] featureColumns = new string[] 
        {
            "DGK",
            "DNK",
            "DBK",
            "АПС"
        //"ПС",
        //"БК",
        //"ГК",
        //"НКТб",
        //"ГГКП",
         };

        /// <summary>
        /// Искомый столбец
        /// </summary>
        private string[] target = new string[] { "Интерпретация коллектора" };

        /// <summary>
        /// Установка гиперпараметров для модели
        /// </summary>
        /// <param name="paramSet"> Набор гиперпараметров в виде словаря с парами (имя параметра, значение) </param>
        public void SetParams(HyperparamSet paramSet)
        {
            this.set = paramSet;
        }

        /// <summary>
        /// Строим модель - здесь происходит добавление слоев
        /// + объявляется оптимизатор, для которого мы устанавливаем скорость обучения (пока не работает из-за ошибки)
        /// </summary>
        /// <param name="complexFinal"> Датасет, передается по сути просто для того чтобы заглянуть в его размерность </param>
        /// <returns> Возвращает модель </returns>
        public Sequential Build(NDarray complexFinal)
        {
            this.model.Add(new LSTM(units: 50, activation: "tanh", return_sequences: true, input_shape: (this.set.SequenceLength, complexFinal.shape[2])));
            for (int i = 0; i < this.set.Layers; i++)
            {
                this.model.Add(new LSTM(units: this.set.Neurons, activation: "tanh", return_sequences: true));
            }

            this.model.Add(new Dense(units: 1, activation: "sigmoid"));

            ////var optimizer = new Keras.Optimizers.Adam(lr: (float)learningRate); 

            this.model.Compile(optimizer: "adam", loss: "binary_crossentropy", metrics: new string[] { "accuracy" });
            return this.model;
        }

        /// <summary>
        /// Вызов Fit, тренировка модели. Пока нигде не используется
        /// </summary>
        /// <param name="complexFinalArray"> Набор признаков для обучения </param>
        /// <param name="parameterFinalArray"> Целевая переменная для обучения </param>
        /// <param name="batchSize"> Размер пакета </param>
        public void Fit(NDarray complexFinalArray, NDarray parameterFinalArray, int batchSize)
        {
            this.model.Fit(complexFinalArray, parameterFinalArray, epochs: this.set.Epochs, batch_size: batchSize);
        }
        
        /// <summary>
        /// Чтение из файла, разделение на таблицы целевой переменной и признаков
        /// </summary>
        /// <param name="path"> путь к файлу </param>
        /// <param name="complex"> таблица признаков </param>
        /// <param name="parameter"> таблица целевой переменной </param>
        public void ReadData(string path, out DataTable complex, out DataTable parameter)
        {
            var data = this.ReadDataFromFile(path);
            data = this.GetFeaturesFromDataTable(data, this.featureColumns.Concat(this.target).ToArray() );
            data = this.DropNaN(data);
            complex = this.GetFeaturesFromDataTable(data, this.featureColumns);
            parameter = this.GetFeaturesFromDataTable(data, this.target);
        }

        /// <summary>
        /// Разбиение на обучающую и тестовую выборки
        /// </summary>
        /// <param name="complex"> таблица с признаками </param>
        /// <param name="parameter"> таблица с целевой переменной </param>
        /// <param name="complexTrainDT"> обучающая таблица с признаками </param>
        /// <param name="parameterTrainDT"> обучающая таблица с целевой переменной </param>
        /// <param name="complexTestDT"> тестовая таблица с признаками </param>
        /// <param name="parameterTestDT"> тестовая таблица с целевой переменной </param>
        public void TrainTestSplit(DataTable complex, DataTable parameter, out DataTable complexTrainDT, out DataTable parameterTrainDT, out DataTable complexTestDT, out DataTable parameterTestDT)
        {
            Console.WriteLine("1.5 Определение тренировочного и тестового массивов данных");

            var totalRows = complex.Rows.Count;
            var trainSize = (int)(totalRows * 0.7);

            complexTrainDT = complex.AsEnumerable().Take(trainSize).CopyToDataTable();
            parameterTrainDT = parameter.AsEnumerable().Take(trainSize).CopyToDataTable();
            complexTestDT = complex.AsEnumerable().Skip(trainSize).Take(totalRows - trainSize).CopyToDataTable();
            parameterTestDT = parameter.AsEnumerable().Skip(trainSize).Take(totalRows - trainSize).CopyToDataTable();
        }

        /// <summary>
        /// Перевод данных в NDarray и resize массивов
        /// </summary>
        /// <param name="complexTrainDT"> обучающая таблица с признаками </param>
        /// <param name="parameterTrainDT"> обучающая таблица с целевой переменной </param>
        /// <param name="complexTestDT"> тестовая таблица с признаками </param>
        /// <param name="parameterTestDT"> тестовая таблица с целевой переменной </param>
        /// <param name="complexTrainFinal"> итоговый обучающий массив признаков </param>
        /// <param name="complexTestFinal"> итоговый тестовый массив признаков </param>
        /// <param name="parameterTrainFinal"> итоговый обучающий массив целевой переменной </param>
        /// <param name="parameterTestFinal"> итоговый тестовый массив целевой переменной </param>
        public void PrepareData(DataTable complexTrainDT, DataTable parameterTrainDT, DataTable complexTestDT, DataTable parameterTestDT, out NDarray complexTrainFinal, out NDarray complexTestFinal, out NDarray parameterTrainFinal, out NDarray parameterTestFinal)
        {
            var complexTrain = this.TableToArray(complexTrainDT);
            var parameterTrain = this.TableToArray(parameterTrainDT);
            var complexTest = this.TableToArray(complexTestDT);
            var parameterTest = this.TableToArray(parameterTestDT);

            if (this.isWindowing)
            {
                this.Windowing(complexTrain, parameterTrain, complexTest, parameterTest, out complexTrainFinal, out complexTestFinal, out parameterTrainFinal, out parameterTestFinal);
            }
            else
            {
                this.TrainTestResize(complexTrain, parameterTrain, complexTest, parameterTest, out complexTrainFinal, out complexTestFinal, out parameterTrainFinal, out parameterTestFinal);
            }
        }

        /// <summary>
        /// Получение предсказаний
        /// В зависимости от того, используется ли оконный метод, происходят (или не происходят) преобразования набора данных
        /// </summary>
        /// <param name="complexTest">  набор данных для которого нужно получить предсказания </param>
        /// <param name="probabilities"> массив вероятностей </param>
        /// <returns> Возвращает предсказания и массив вероятностей </returns>
        public NDarray Predict(NDarray complexTest, out NDarray probabilities)
        {
            probabilities = this.model.Predict(complexTest);
            var predictedValues = np.around(this.model.Predict(complexTest), 0);

            if (this.isWindowing)
            {
                var firstBatch = np.resize(predictedValues, new Shape(predictedValues.shape[1], 1));
                var size = predictedValues.shape[0] * predictedValues.shape[1];
                var stackedPredictValues = np.resize(predictedValues, new Shape(size, 1));
                stackedPredictValues = stackedPredictValues[new Slice(0, null, this.set.SequenceLength)];
                stackedPredictValues = np.delete(stackedPredictValues, 0, axis: 0);
                predictedValues = np.concatenate(new NDarray[] { firstBatch, stackedPredictValues });

                var firstBatchProbabilities = np.resize(probabilities, new Shape(probabilities.shape[1], 1));
                var sizeProbabilities = probabilities.shape[0] * probabilities.shape[1];
                var stackedProbabilities = np.resize(probabilities, new Shape(sizeProbabilities, 1));
                stackedProbabilities = stackedProbabilities[new Slice(0, null, this.set.SequenceLength)];
                stackedProbabilities = np.delete(stackedProbabilities, 0, axis: 0);
                probabilities = np.concatenate(new NDarray[] { firstBatchProbabilities, stackedProbabilities });
            }
            else
            {
                predictedValues = np.resize(predictedValues, new Shape(predictedValues.shape[0], 1));
                probabilities = np.resize(probabilities, new Shape(probabilities.shape[0], 1));
            }

            Console.WriteLine("!Тестирование завершено");

            return predictedValues;
        }
        
        /// <summary>
        /// Подсчёт точности по формуле (число правильных предсказаний / число предсказаний) 
        /// Вместо этой функции можно использовать метод Evaluate из Keras, но почему-то он не работает
        /// </summary>
        /// <param name="predictedValues"> полученные предсказания </param>
        /// <param name="parameterTest"> правильные ответы </param>
        /// <returns> Процент правильных предсказаний </returns>
        public float AccuracyScore(NDarray predictedValues, NDarray parameterTest)
        {
            var accuracy = 0f;
            if (this.isWindowing)
            {
                for (var i = 0; i < predictedValues.size; i++)
                {
                    CultureInfo ci = (CultureInfo)CultureInfo.CurrentCulture.Clone();
                    ci.NumberFormat.CurrencyDecimalSeparator = ".";
                    var predictedValue = (int)float.Parse(predictedValues[i][0].repr, NumberStyles.Any, ci);
                    var actualValue = (int)float.Parse(parameterTest[i][0][0].repr, NumberStyles.Any, ci);
                    if (predictedValue == actualValue)
                    {
                        accuracy += 1;
                    }
                }

                accuracy = accuracy / predictedValues.size;
                return accuracy;
            }
            else
            {
                for (var i = 0; i < predictedValues.size; i++)
                {
                    CultureInfo ci = (CultureInfo)CultureInfo.CurrentCulture.Clone();
                    ci.NumberFormat.CurrencyDecimalSeparator = ".";
                    var predictedValue = (int)float.Parse(predictedValues[i][0].repr, NumberStyles.Any, ci);
                    var actualValue = (int)float.Parse(parameterTest[i][0].repr, NumberStyles.Any, ci);
                    if (predictedValue == actualValue)
                    {
                        accuracy += 1;
                    }
                }

                accuracy = accuracy / predictedValues.size;
                return accuracy;
            }
        }

        /// <summary>
        /// Генерация наборов гиперпараметров
        /// </summary>
        /// <param name="selectionCount"> Количество наборов </param>
        /// <returns> Возвращает список словарей с парами (имя параметра, значение) </returns>
        public List<HyperparamSet> GenerateParamSets(int selectionCount)
        {
            Random rnd = new Random(0);

            List<HyperparamSet> paramSets = new List<HyperparamSet>();
            for (var i = 0; i < selectionCount; i++)
            {
                var paramSet = new HyperparamSet(
                    sequenceLength: rnd.Next(1, 6),
                    step: 1,
                    layers: rnd.Next(1, 4),
                    neurons: rnd.Next(5, 10),
                    learningRate: (float)((rnd.NextDouble() * (0.1 - 0.001)) + 0.001),
                    epochs: rnd.Next(50, 201));

                paramSets.Add(paramSet);
            }

            return paramSets;
        }

        /// <summary>
        /// Поиск лучших гиперпараметров
        /// Генерируем заданное количество наборов гиперпараметров
        /// Для каждого набора производим кросс-валидацию в numFolds фолдов
        /// </summary>
        /// <param name="complex"> Набор признаков </param>
        /// <param name="parameter"> Целевая переменная </param>
        /// <param name="numFolds"> Количество фолдов для кросс-валидации </param>
        /// <param name="setsCount"> Количество наборов гиперпараметров которое нужно сгенерировать </param>
        /// <returns> возвращает лучший набор гиперпараметров </returns>
        public HyperparamSet FindHyperparams(DataTable complex, DataTable parameter, int numFolds, int setsCount)
        {
            var paramSets = this.GenerateParamSets(setsCount);
            var bestAccuracy = 0f;
            HyperparamSet bestSet = null;
            foreach (var set in paramSets)
            {
                this.SetParams(set);
                var setResult = this.CrossValidation(complex, parameter, numFolds);
                if (bestAccuracy < setResult)
                {
                    bestAccuracy = setResult;
                    bestSet = set;
                }

                this.model = new Sequential();
            }

            return bestSet;
        }

        /// <summary>
        /// Кросс-валидация: происходит разбиение на обучающую и тестовую выборки с учетом фолдов, затем происходит обучение модели и предсказание
        /// </summary>
        /// <param name="complex"> Набор признаков </param>
        /// <param name="parameter"> Целевая переменная </param>
        /// <param name="numFolds"> Количество фолдов </param>
        /// <returns> Возвращает среднюю точность по фолдам для заданного набора гиперпараметров </returns>
        public float CrossValidation(DataTable complex, DataTable parameter, int numFolds)
        {
            var totalRows = complex.Rows.Count;
            var foldSize = totalRows / numFolds;
            List<float> foldsAccuracy = new List<float>();
            for (int fold = 1; fold <= numFolds; fold++)
            {
                var startIndex = (fold - 1) * foldSize;
                var endIndex = fold * foldSize;
                if (fold == numFolds)
                {
                    endIndex = totalRows;
                }

                var complexTrainDT = this.ExtractSubset(complex, 0, startIndex)
                    .AsEnumerable()
                    .Union(this.ExtractSubset(complex, endIndex, totalRows - endIndex).AsEnumerable())
                    .CopyToDataTable();

                var parameterTrainDT = this.ExtractSubset(parameter, 0, startIndex)
                    .AsEnumerable()
                    .Union(this.ExtractSubset(parameter, endIndex, totalRows - endIndex).AsEnumerable())
                    .CopyToDataTable();

                var complexTestDT = this.ExtractSubset(complex, startIndex, endIndex - startIndex);
                var parameterTestDT = this.ExtractSubset(parameter, startIndex, endIndex - startIndex);

                NDarray complexFinal;
                NDarray complexTestFinal;
                NDarray parameterTrainingFinal;
                NDarray parameterTestFinal;

                this.PrepareData(complexTrainDT, parameterTrainDT, complexTestDT, parameterTestDT, out complexFinal, out complexTestFinal, out parameterTrainingFinal, out parameterTestFinal);
                this.Build(complexFinal);
                this.Fit(complexFinal, parameterTrainingFinal, this.batchSize);
                NDarray probabilities;
                var predictedValues = this.Predict(complexTestFinal, out probabilities);
                var probabilitiesFinal = ConvertProbabilities(probabilities);
                var accuracy = this.AccuracyScore(predictedValues, parameterTestFinal);
                Console.WriteLine(accuracy);
                foldsAccuracy.Add(accuracy); 
            }

            return foldsAccuracy.Average();
        }

        /// <summary>
        /// Конвертация NDarray вероятностей в массив double 
        /// </summary>
        /// <param name="probabilities"> NDarray вероятностей </param>
        /// <returns> [][] массив double вероятностей </returns>
        public double[][] ConvertProbabilities(NDarray probabilities)
        {
            int numSamples = probabilities.shape[0];

            var result = new double[numSamples][];
            for (int i = 0; i < numSamples; i++)
            {
                CultureInfo ci = (CultureInfo)CultureInfo.CurrentCulture.Clone();
                ci.NumberFormat.CurrencyDecimalSeparator = ".";
                //double prob = probabilities[i, j, 0];
                double prob = float.Parse(probabilities[i, 0].repr, NumberStyles.Any, ci);
                result[i] = new double[2] { prob, 1 - prob };
            }

            return result;
        }

        /// <summary>
        /// Нужна чтобы вытаскивать фрагменты из DataTable
        /// </summary>
        /// <param name="table"> таблица DataTable </param>
        /// <param name="startIndex"> индекс с которого нужно вытаскивать </param>
        /// <param name="count"> Сколько строк нужно </param>
        /// <returns> Возвращает таблицу DataTable из определенного числа строк начиная со стартового индекса </returns>
        public DataTable ExtractSubset(DataTable table, int startIndex, int count)
        {
            DataTable subset = table.Clone(); 
            for (int i = startIndex; i < startIndex + count; i++)
            {
                DataRow row = table.Rows[i];
                subset.ImportRow(row);
            }

            return subset;
        }

        /// <summary>
        /// Функция преобразования DataTable в NDarray
        /// </summary>
        /// <param name="table"> таблица DataTable </param>
        /// <returns> Возвращает NDarray конвертированный из таблицы </returns>
        public NDarray TableToArray(DataTable table)
        {
            List<float[]> myTable = new List<float[]>();
            foreach (DataRow dr in table.Rows)
            {
                var columnCount = 0;
                float[] myTableRow = new float[table.Columns.Count];
                foreach (var item in dr.ItemArray)
                { 
                    CultureInfo ci = (CultureInfo)CultureInfo.CurrentCulture.Clone();
                    ci.NumberFormat.CurrencyDecimalSeparator = ".";
                    myTableRow[columnCount] = float.Parse(item.ToString(), NumberStyles.Any, ci);
                    columnCount++;
                }

                myTable.Add(myTableRow);
            }

            var table2DArray = myTable.ToArray();

            var numRows = table.Rows.Count;
            var numColumns = table.Columns.Count;

            float[,] dataArray = new float[numRows, numColumns];
            for (int row = 0; row < numRows; row++)
            {
                for (int col = 0; col < numColumns; col++)
                {
                    dataArray[row, col] = table2DArray[row][col];
                }
            }

            NDarray ndarray = np.array(dataArray);

            return ndarray;
        }

        /// <summary>
        /// Чтение DataTable из Txt файла
        /// </summary>
        /// <param name="filePath"> путь к файлу </param>
        /// <returns> контент файла в DataTable </returns>
        public DataTable ReadDataFromFile(string filePath)
        {
            DataTable data = new DataTable();

            using (StreamReader reader = new StreamReader(filePath))
            {
                var line = reader.ReadLine();

                if (!string.IsNullOrEmpty(line))
                {
                    string[] headers = line.Split('\t');

                    foreach (string header in headers)
                    {
                        data.Columns.Add(header);
                    }

                    while ((line = reader.ReadLine()) != null)
                    {
                        string[] values = line.Split('\t');
                        var row = data.NewRow();
                        row.ItemArray = values;
                        data.Rows.Add(row);
                    }
                }
            }

            return data;
        }

        /// <summary>
        /// Избавиться от NaN в таблице
        /// </summary>
        /// <param name="data"> таблица DataTable </param>
        /// <returns> Возвращает таблицу без NaN значений </returns>
        public DataTable DropNaN(DataTable data)
        {
            for (int i = data.Rows.Count - 1; i >= 0; i--)
            {
                DataRow row = data.Rows[i];
                if (row.ItemArray.Any(item => item is DBNull))
                {
                    row.Delete();
                }
                else if (row.ItemArray.Any(item => item is ""))
                {
                    row.Delete();
                }
            }

            data.AcceptChanges();
            return data;
        }

        /// <summary>
        /// Функция для вытаскивания колонок из DataTable
        /// </summary>
        /// <param name="data"> таблица DataTable </param>
        /// <param name="featureColumns"> массив названий колонок которые нужно получить </param>
        /// <returns> DataTable с необходимыми столбцами </returns>
        public DataTable GetFeaturesFromDataTable(DataTable data, string[] featureColumns)
        {
            DataTable features = new DataView(data).ToTable(false, featureColumns);

            return features;
        }

        /// <summary>
        /// Использование оконного метода
        /// </summary>
        /// <param name="complexTrain"> обучающая выборка признаки</param>
        /// <param name="parameterTrain"> обучающая выборка целевая переменная </param>
        /// <param name="complexTest"> тестовая выборка признаки </param>
        /// <param name="parameterTest"> тестовая выборка целевая переменная </param>
        /// <param name="complexFinalTrainingArray"> выходной обучающий массив признаков </param>
        /// <param name="complexFinalTestArray"> выходной тестовый массив признаков </param>
        /// <param name="parameterTrainingFinal"> выходной обучающий массив целевой переменной </param>
        /// <param name="parameterTestFinal"> выходной тестовый массив целевой переменной </param>
        private void Windowing(NDarray complexTrain, NDarray parameterTrain, NDarray complexTest, NDarray parameterTest, out NDarray complexFinalTrainingArray, out NDarray complexFinalTestArray, out NDarray parameterTrainingFinal, out NDarray parameterTestFinal)
        {
            complexTrain = np.resize(complexTrain, new Shape(complexTrain.shape[0], this.set.Step, complexTrain.shape[1]));
            List<NDarray> complexFinal = new List<NDarray>();
            for (int i = this.set.SequenceLength; i <= complexTrain.shape[0]; i++)
            {
                complexFinal.Add(complexTrain[$"{i - set.SequenceLength}:{i}", 0]);
            }

            complexFinalTrainingArray = np.array(complexFinal);

            complexTest = np.resize(complexTest, new Shape(complexTest.shape[0], this.set.Step, complexTest.shape[1]));
            List<NDarray> complexTestFinal = new List<NDarray>();
            for (int i = this.set.SequenceLength; i <= complexTest.shape[0]; i++)
            {
                complexTestFinal.Add(complexTest[$"{i - set.SequenceLength}:{i}", 0]);
            }

            complexFinalTestArray = np.array(complexTestFinal);

            parameterTrain = np.resize(parameterTrain, new Shape(parameterTrain.shape[0], this.set.Step, parameterTrain.shape[1]));
            List<NDarray> parameterFinal = new List<NDarray>();
            for (int i = this.set.SequenceLength; i <= parameterTrain.shape[0]; i++)
            {
                parameterFinal.Add(parameterTrain[$"{i - this.set.SequenceLength}:{i}", 0]);
            }

            parameterTrainingFinal = np.array(parameterFinal);

            parameterTest = np.resize(parameterTest, new Shape(parameterTest.shape[0], this.set.Step, parameterTest.shape[1]));
            List<NDarray> parameterFinalTest = new List<NDarray>();
            for (int i = this.set.SequenceLength; i <= parameterTest.shape[0]; i++)
            {
                parameterFinalTest.Add(parameterTest[$"{i - this.set.SequenceLength}:{i}", 0]);
            }

            var parameterTestFinalArray = np.array(parameterFinalTest);

            parameterTestFinal = parameterTest;

            Console.WriteLine($"Обучающая: {complexFinalTrainingArray.shape}");
            Console.WriteLine($"Тестовая:  {complexFinalTestArray.shape}");
            Console.WriteLine($"Target (train), {parameterTrainingFinal.shape}");
            Console.WriteLine($"Target (test), {parameterTestFinal.shape}");
        }

        /// <summary>
        /// Resize который происходит вместо использования оконного метода
        /// </summary>
        /// <param name="complexTrain"> обучающий набор признаков </param>
        /// <param name="parameterTrain"> целевая переменная для обучения </param>
        /// <param name="complexTest"> тестовый набор признаков </param>
        /// <param name="parameterTest"> ответы для тестового набора </param> 
        /// <param name="complexTrainingFinal"> итоговый обучающий массив признаков </param>
        /// <param name="complexTestFinal"> итоговый тестовый массив признаков </param>
        /// <param name="parameterTrainingFinal"> итоговый обучающий массив целевой переменной </param>
        /// <param name="parameterTestFinal"> итоговый тестовый массив целевой переменной </param>
        private void TrainTestResize(NDarray complexTrain, NDarray parameterTrain, NDarray complexTest, NDarray parameterTest, out NDarray complexTrainingFinal, out NDarray complexTestFinal, out NDarray parameterTrainingFinal, out NDarray parameterTestFinal)
        {
            complexTrainingFinal = np.resize(complexTrain, new Shape(complexTrain.shape[0], this.set.SequenceLength, complexTrain.shape[1]));
            complexTestFinal = np.resize(complexTest, new Shape(complexTest.shape[0], this.set.SequenceLength, complexTest.shape[1]));
            parameterTrainingFinal = np.resize(parameterTrain, new Shape(parameterTrain.shape[0], this.set.SequenceLength, parameterTrain.shape[1]));
            parameterTestFinal = parameterTest;
            Console.WriteLine($"Обучающая: {complexTrainingFinal.shape}");
            Console.WriteLine($"Тестовая: {complexTestFinal.shape}");
            Console.WriteLine($"Target (train), {parameterTrainingFinal.shape}");
            Console.WriteLine($"Target (test), {parameterTest.shape}");
        }
    }
}
