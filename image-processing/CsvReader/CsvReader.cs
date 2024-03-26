using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using Microsoft.VisualBasic.FileIO;

public static class CsvReader<T>
{
    public static T[] ReadRow(string path, uint targetLine)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException("File does not exist.", path);

        TextFieldParser parser = new TextFieldParser(path) { TextFieldType = FieldType.Delimited };
        parser.SetDelimiters(",");

        int count = 1;

        while (!parser.EndOfData)
        {
            string[] row = parser.ReadFields();
            if (count == targetLine)
            {
                var size = row.Length;
                var vector = new T[size];
                for (int i = 0; i < size; i++)
                    vector[i] = (T)Convert.ChangeType(row[i], typeof(T));
                
                return vector;
            }

            count++;
        }

        return new T[0];
    }

    public static void SaveToJson(T[] row)
    {
        string json = JsonSerializer.Serialize<IEnumerable<T>>(row);
        File.WriteAllText("Rows.json", json);
    }
}
