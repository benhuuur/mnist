using System;
using System.Windows.Forms;
using ImageProcessing;

var path = "../../data/mnist_train.csv";

try
{
for (uint i = 240; i < 240 + 10; i++)
{
    var row = CsvReader<byte>.ReadRow(path, i);

    Console.WriteLine(row[0]);

    CsvReader<byte>.SaveToJson(row);
    ImageGenerator.GenerateImageFromGrayVector(row, name: $"../../data/images/csv{i}.png");
}

//     var array = ImageGenerator.GetVectorFromImage(@"..\..\data\images\test.png");
//     CsvReader<byte>.SaveToJson(array);
//     ImageGenerator.GenerateImageFromGrayVector(array);

//     MessageBox.Show("top");
}
catch (Exception ex)
{
    MessageBox.Show(ex.Message);
}
