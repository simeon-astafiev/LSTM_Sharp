using Numpy;
using System;
using System.ComponentModel;
using System.Data;
using System.Windows.Forms;

namespace LSTM_Sharp
{
    /// <summary>
    /// Класс формы
    /// </summary>
    public partial class Form1 : Form
    {
        /// <summary>
        /// Инициализация
        /// </summary>
        public Form1()
        {
            InitializeComponent();
        }

        /// <summary>
        /// Нажатие на кнопку выбора файла
        /// </summary>
        /// <param name="sender"> sender </param>
        /// <param name="e"> event </param>
        private void button2_Click(object sender, EventArgs e)
        {
            openFileDialog1.ShowDialog();
        }

        /// <summary>
        /// Подтверждение выбора файла
        /// </summary>
        /// <param name="sender"> sender </param>
        /// <param name="e"> event </param>
        private void openFileDialog1_FileOk(object sender, CancelEventArgs e)
        {
            var file = openFileDialog1.FileName;
            textBox1.Text = file;
            button1.Enabled = true;
        }

        /// <summary>
        /// Нажатие на кнопку старт
        /// </summary>
        /// <param name="sender"> sender r</param>
        /// <param name="e"> even t</param>
        private void button1_Click(object sender, EventArgs e)
        {
            var file = textBox1.Text;
            var model = new Model();

            ////Здесь тест обычного обучения без кросс-валидации

            DataTable complex;
            DataTable parameter;
            model.ReadData(file, out complex, out parameter);
            model.TrainTestSplit(complex, parameter,
                out DataTable Complex_trainDT, out DataTable Parameter_trainDT, out DataTable Complex_testDT, out DataTable Parameter_testDT);
            NDarray Complex_train;
            NDarray Parameter_train;
            NDarray Complex_test;
            NDarray Parameter_test;
            model.PrepareData(Complex_trainDT, Parameter_trainDT, Complex_testDT, Parameter_testDT,
                out Complex_train, out Complex_test, out Parameter_train, out Parameter_test);
            model.Build(Complex_train);
            model.Fit(Complex_train, Parameter_train, 32);
            NDarray probabilities;
            var predictedValues = model.Predict(Complex_test, out probabilities);
            Console.WriteLine(model.AccuracyScore(predictedValues, Parameter_test));

            ////Здесь тест кросс-валидации
            //DataTable complex;
            //DataTable parameter;
            //model.ReadData(file, out complex, out parameter);
            //var bestParams = model.FindHyperparams(complex, parameter, 2, 2);
        }
    }
}
