using Microsoft.ML;
using System;
using System.Text;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;

namespace Vicodin
{
    public partial class MainMenu : Form
    {
        MLContext mlContext;
        public MainMenu()
        {
            mlContext = new MLContext();
            InitializeComponent();
        }

        private void button2_Click(object sender, EventArgs e)
        {
            if(!File.Exists("Data/model.zip"))
            {
                MessageBox.Show("Train model first!");
                return;
            }
            Visible = false;
            TestingForm testingForm = new TestingForm(this, mlContext);
            
        }

        private void button1_Click(object sender, EventArgs e)
        {
            Visible = false;
            TrainingForm trainingForm = new TrainingForm(this, mlContext);
            
        }
    }
}
