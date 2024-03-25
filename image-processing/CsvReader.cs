using System;
using System.Collections.Generic;
using System.IO;
using Microsoft.VisualBasic.FileIO;

public static class CsvReader
{
    private static List<string[]> Rows;
    public static List<string[]> getRows => Rows;

    public static void Read(string Path)
    {
        if (!File.Exists(Path))
        {
            Console.WriteLine("File does not exist.");
            return;
        }

        TextFieldParser parser = new TextFieldParser(Path);
        parser.TextFieldType = FieldType.Delimited;
        parser.SetDelimiters(",");
        while (!parser.EndOfData)
        {
            string[] row = parser.ReadFields();
            Rows.Add(row);
        }
    }
}
