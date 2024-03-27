using System;
using System.Windows.Forms;
using ImageProcessing;

// var path = "mnist_train.csv";
// var row = CsvReader<byte>.ReadRow(path, 232);

// Console.WriteLine(row[0]);

// CsvReader<byte>.SaveToJson(row);
// ImageGenerator.GenerateImageFromGrayVector(row);

try
{
    var array = ImageGenerator.GetVectorFromImage(@"..\data\images\test.png");
    CsvReader<byte>.SaveToJson(array);
    ImageGenerator.GenerateImageFromGrayVector(array);
}
catch (Exception ex)
{
    MessageBox.Show(ex.Message);
}