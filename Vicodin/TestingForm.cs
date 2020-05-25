using System;
using System.Collections.Generic;
using System.Drawing;
using System.Windows.Forms;
using Microsoft.ML;

namespace Vicodin
{
    public partial class TestingForm : Form
    {
        bool isDown;
        MLContext mlContext;
        Form MainMenu;
        PredictionEngine<InputData, OutputData> predictionEngine;
        public TestingForm(Form mainMenu, MLContext mlContext)
        {
            this.mlContext = mlContext;
            MainMenu = mainMenu;
            ITransformer trainedModel = mlContext.Model.Load("Data/Model.zip", out var modelInputSchema);
            predictionEngine = mlContext.Model.CreatePredictionEngine<InputData, OutputData>(trainedModel);
            Visible = true;
            InitializeComponent();

        }

        private void pictureBox1_MouseMove(object sender, MouseEventArgs e)
        {
            if(!isDown)
            {
                return;
            }
            Bitmap b = new Bitmap(pictureBox1.Image);
            Graphics g = Graphics.FromImage(b);
            g.DrawEllipse(new Pen(Color.White, 10), e.X - 3, e.Y - 3, 6, 6);
            pictureBox1.Image = b;
            Predict(new Bitmap(b, new Size(28, 28)));
        }

        private void pictureBox1_MouseUp(object sender, MouseEventArgs e)
        {
            isDown = false;
        }

        private void pictureBox1_MouseDown(object sender, MouseEventArgs e)
        {
            isDown = true;
        }

        private void button1_Click(object sender, EventArgs e)
        {
            pictureBox1.Image = Properties.Resources.Background;
        }

        void Predict(Bitmap b)
        {
            List<float> features = new List<float>();
            for(int i = 0; i < 28; i++)
            {
                for(int j = 0; j < 28; j++)
                {
                    features.Add(b.GetPixel(j, i).R);
                }
            }
            OutputData result = predictionEngine.Predict(new InputData(features.ToArray()));
            float maxVal = 0;
            int label = 0;
            for(int i = 0; i < 10; i++)
            {
                if(result.Score[i] > maxVal)
                {
                    maxVal = result.Score[i];
                    label = i;
                }
            }
            label1.Text = label.ToString();
        }

        private void TestingForm_FormClosed(object sender, FormClosedEventArgs e)
        {
            MainMenu.Visible = true;
        }
    }
}
