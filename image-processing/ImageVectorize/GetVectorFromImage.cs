using System;
using System.Drawing;
using System.IO;

namespace ImageProcessing;

public static partial class ImageGenerator
{
    public static byte[] GetVectorFromImage(string path)
    {
        int index = 0;
        Bitmap image = new Bitmap(path);
        int height = image.Height;
        int width = image.Width;

        var grayVector = new byte[height * width];

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                Color pixel = image.GetPixel(x, y);
                byte intensity = (byte)(255 - (0.299 * pixel.R + 0.587 * pixel.G + 0.114 * pixel.B));
                grayVector[index] = intensity;
                index++;
            }
        }

        return grayVector;
    }
}
