using System;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace Vicodin
{
    public partial class TrainingForm : Form
    {
        MLContext mlContext;
        Form mainMenu;
        public TrainingForm(Form mainMenu, MLContext mlContext)
        {
            this.mlContext = mlContext;
            this.mainMenu = mainMenu;
            Visible = true;
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            richTextBox1.AppendText("Training...");
            button1.Enabled = false;
            Task.Factory.StartNew(() =>
            {
                string result = Train(mlContext);
                richTextBox1.Invoke((MethodInvoker)(() =>
                {
                    richTextBox1.AppendText(result);
                }));
                button1.Invoke((MethodInvoker)(() =>
                {
                    button1.Enabled = true;
                }));
            });    
        }

        public static string Train(MLContext mlContext)
        {
            try
            {
                var trainData = mlContext.Data.LoadFromTextFile(path: "Data/mnist_train.csv",
                        columns: new[]
                        {
                            new TextLoader.Column(nameof(InputData.PixelValues), DataKind.Single, 1, 784),
                            new TextLoader.Column("Number", DataKind.Single, 0)
                        },
                        hasHeader: true,
                        separatorChar: ','
                        );


                var testData = mlContext.Data.LoadFromTextFile(path: "Data/mnist_test.csv",
                        columns: new[]
                        {
                            new TextLoader.Column(nameof(InputData.PixelValues), DataKind.Single, 1, 784),
                            new TextLoader.Column("Number", DataKind.Single, 0)
                        },
                        hasHeader: true,
                        separatorChar: ','
                        );

                var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", "Number", keyOrdinality: ValueToKeyMappingEstimator.KeyOrdinality.ByValue).
                    Append(mlContext.Transforms.Concatenate("Features", nameof(InputData.PixelValues)).AppendCacheCheckpoint(mlContext));

                var trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features");
                var trainingPipeline = dataProcessPipeline.Append(trainer).Append(mlContext.Transforms.Conversion.MapKeyToValue("Number", "Label"));

                ITransformer trainedModel = trainingPipeline.Fit(trainData);

                var predictions = trainedModel.Transform(testData);
                var metrics = mlContext.MulticlassClassification.Evaluate(data: predictions, labelColumnName: "Number", scoreColumnName: "Score");


                mlContext.Model.Save(trainedModel, trainData.Schema, "Data/Model.zip");
                return $"Model trained successfully.\n Accuracy - {metrics.MicroAccuracy}";
            }
            catch(Exception ex)
            {
                return $"Failed to train model due to exception \"{ex.Message}\"";
            }
        }

        private void TrainingForm_FormClosed(object sender, FormClosedEventArgs e)
        {
            mainMenu.Visible = true;
        }
    }
}
