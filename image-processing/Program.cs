using System;
using ImageProcessing;

// var path = "mnist_train.csv";
// var row = CsvReader<byte>.ReadRow(path, 15);

// Console.WriteLine(row[0].GetType());

// CsvReader<byte>.SaveToJson(row);
// ImageGenerator.GenerateImageFromGrayVector(row);

var array = ImageGenerator.GetVectorFromImage("images/test.png");
CsvReader<byte>.SaveToJson(array);
ImageGenerator.GenerateImageFromGrayVector(array);